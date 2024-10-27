import tensorflow as tf
from keras import layers
from transformer import Encoder, EncoderLayer, CrossAttention, TFPositionalEncoding2D
import keras
from math import floor
from conv import ResidualBlock

class PairGenerator(layers.Layer):
  def __init__(self, width, is_first=True, dropout_rate=0.1,vocab_size=21):
    super(PairGenerator, self).__init__()
    self.width = width
    self.dropout = tf.keras.layers.Dropout(dropout_rate)
    if is_first:
      self.conv = keras.layers.Conv1D(width-1,1) 
      self.embed = tf.keras.layers.Embedding(input_dim=vocab_size,output_dim=width-1,mask_zero=True)
      self.posEncode = TFPositionalEncoding2D(width-1)
    else:
      self.conv = keras.layers.Conv1D(width,1) 
    self.dists = None

    self.is_first = is_first

   
  def get_dists(self):
      if self.dists is not None:
         return self.dists
      # do the / 256 to get into range of 0 to 1 
      # do the 1 - dists because it means that the value will decrease the farther it is from each other so the distance weight increases 
      # as it is hard to react to zeros in matrix muls 
      dists = tf.expand_dims(tf.range(256), axis=1) - tf.expand_dims(tf.range(256), axis=0)
      dists = tf.cast(dists,dtype=tf.float32)
      dists = tf.math.log(tf.abs(dists / 256)+1e-8)
      dists = tf.expand_dims(dists, axis=0)
      dists = tf.expand_dims(dists, axis=-1)

      self.dists = dists
      return self.dists
  
  def call(self,context,proc_context=None,training=False):
    if self.is_first:
      if isinstance(context, dict):
         context = context['primary_onehot']
      context = tf.argmax(context, axis=-1)
      context = self.embed(context) + self.conv(proc_context)
    else:
      context = self.conv(context)
    x1 = tf.expand_dims(context, axis=2)  # Shape becomes (batch_size, tokens, 1, rep_size)
    x2 = tf.expand_dims(context, axis=1)  # Shape becomes (batch_size, 1, tokens, rep_size)
    tokens = context.shape[1]
   
    # used add instead of concat because this means that Xij and Xji are treated the same
    pairwise = tf.tile(x1, [1, 1, tokens, 1]) + tf.tile(x2, [1, tokens, 1, 1])

    
    if self.is_first:
      pairwise = self.posEncode(pairwise)
      dists = tf.tile(self.get_dists(), [pairwise.shape[0], 1, 1, 1])
      pairwise = tf.concat([pairwise, dists], axis=-1)

    pairwise = self.dropout(pairwise)
    return pairwise

class CrossLayer(layers.Layer): # don't use real cross attention as is too memory intensive and can give information about layer
   def __init__(self,width):
      super(CrossLayer, self).__init__()
      # self.pairGen = PairGenerator(width)
      self.width = width
      self.pair = PairGenerator(width,is_first=False)
      

   def get_config(self):
        config = super(CrossLayer, self).get_config()
        config.update({"width": self.width})
        return config

   def call(self, x, context):
      pair = self.pair(context)

      if x.shape[-2] != pair.shape[-2]:
         pool = pair.shape[-2] / x.shape[-2]
         pair = tf.nn.max_pool2d(pair, pool, pool, padding='SAME')

      return x + pair

class BackwardCrossLayer(layers.Layer):
   def __init__(self, width,dimension,dropout_rate=0.1):
      super(BackwardCrossLayer, self).__init__()
      #self.dense_layers = [tf.keras.layers.Dense(8) for _ in range(256)]

      self.cross = CrossAttention(num_heads=2,key_dim=dimension)
      self.conv = keras.layers.Conv1D(1,1)
      self.width = width
      self.dropout = tf.keras.layers.Dropout(dropout_rate)
   
   def get_config(self):
        config = super(BackwardCrossLayer, self).get_config()
        config.update({"width": self.width})
        return config

   def call(self, x, context):
      #output_rows = [self.dense_layers[i](x[0][i]) for i in range(256)]
      flat = tf.squeeze(self.conv(x),axis=-1)
      flat = keras.activations.relu(flat,alpha=0.1)
      flat = self.dropout(flat)
      return self.cross(context,flat)


class ModelLayer(layers.Layer):
   def __init__(self, width, dimension, change=0, dropout_rate=0.1):
      super(ModelLayer, self).__init__()
      
      self.width = width
      self.dimension = dimension

      self.blocks = [ResidualBlock(width) for _ in range(3)] # don't use attention as takes up too much memory

      self.encoderLayers = [EncoderLayer(d_model=dimension,num_heads=2,dff=dimension+2) for _ in range(1)]

      self.cross = CrossLayer(width)

      self.dropout = tf.keras.layers.Dropout(dropout_rate)

      self.change = change

   def get_config(self):
      config = super(ModelLayer, self).get_config()
      config.update({
         "width": self.width,
         'dimension': self.dimension
         })
      return config

   def call(self, x, context, skips=[]):

      for i in range(len(self.blocks)):
        block = self.blocks[i]
        x = block(x)
        x = (x + tf.transpose(x, perm=[0, 2, 1, 3]))/2.0 # inforce Xij = Xji
        if self.change > 0 and i < 1: #or (i == 0 and self.change == 0):
           skips.append(x)
           x = tf.nn.max_pool2d(x,2,2,padding='SAME')
        elif self.change < 0 and i < 1: #or (i == 2 and self.change == 0):
           x = keras.layers.UpSampling2D()(x)
           x += skips.pop()

      #context = self.backward(x, context)
      for encode in self.encoderLayers:
        context = encode(context)
      
      x = self.cross(x,context)

      return x, context



class Model(keras.Model):
    def __init__(self, dimension=18,width=6):
        super().__init__()

        self.dimension = dimension
        self.width = width
        
        self.encoder = Encoder(num_heads=2,num_layers=2,dff=dimension+2,d_model=dimension) # use doble headed to show I can

        self.pairGen = PairGenerator(width=width)

        self.modelLayers = [ModelLayer(width,dimension,change=c) for c in [1,0,-1]]

        self.outConv = layers.Conv2D(1,3,padding='SAME')

    def get_config(self):
        config = super(Model, self).get_config()
        config.update({"dimension": self.dimension,
                       "width": self.width})
        return config

    def call(self, context):
       org_context = context
       context = self.encoder(context)
       x = self.pairGen(org_context,context)

       for layer in self.modelLayers:
          x, context = layer(x, context)
       

       x = (x + tf.transpose(x, perm=[0, 2, 1, 3]))/2.0 # inforce Xij = Xji
       x = self.outConv(x)
       # as distances can't be neggitve
       x = tf.math.softplus(x)
       return x
       
       