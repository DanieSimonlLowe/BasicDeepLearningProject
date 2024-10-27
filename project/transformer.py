import tensorflow as tf
from keras import layers
import keras
import numpy as np
import math

#from conv import sinusoidal_embedding



def positional_encoding(length, depth):
        # https://www.tensorflow.org/text/tutorials/transformer
        depth = depth/2

        positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
        depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)

        angle_rates = 1 / (10000**depths)         # (1, depth)
        angle_rads = positions * angle_rates      # (pos, depth)

        pos_encoding = np.concatenate(
            [np.sin(angle_rads), np.cos(angle_rads)],
            axis=-1) 

        return tf.cast(pos_encoding, dtype=tf.float32)


def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = tf.stack((tf.sin(sin_inp), tf.cos(sin_inp)), -1)
    emb = tf.reshape(emb, (*emb.shape[:-2], -1))
    return emb

class TFPositionalEncoding2D(layers.Layer):
    # https://github.com/tatp22/multidim-positional-encoding/blob/master/positional_encodings/tf_encodings.py
    def __init__(self, channels: int, dtype=tf.float32):
        """
        Args:
            channels int: The last dimension of the tensor you want to apply pos emb to.

        Keyword Args:
            dtype: output type of the encodings. Default is "tf.float32".

        """
        super(TFPositionalEncoding2D, self).__init__()

        self.channels = int(2 * np.ceil(channels / 4))
        self.inv_freq = np.float32(
            1
            / np.power(
                10000, np.arange(0, self.channels, 2) / np.float32(self.channels)
            )
        )
        self.cached_penc = None

    def get_config(self):
        return super(TFPositionalEncoding2D, self).get_config()

    @tf.function
    def call(self, inputs):
        """
        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(inputs.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")

        if self.cached_penc is not None and self.cached_penc.shape == inputs.shape:
            return self.cached_penc + inputs

        self.cached_penc = None
        _, x, y, org_channels = inputs.shape

        dtype = self.inv_freq.dtype

        pos_x = tf.range(x, dtype=dtype)
        pos_y = tf.range(y, dtype=dtype)

        sin_inp_x = tf.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = tf.einsum("i,j->ij", pos_y, self.inv_freq)

        emb_x = tf.expand_dims(get_emb(sin_inp_x), 1)
        emb_y = tf.expand_dims(get_emb(sin_inp_y), 0)

        emb_x = tf.tile(emb_x, (1, y, 1))
        emb_y = tf.tile(emb_y, (x, 1, 1))
        emb = tf.concat((emb_x, emb_y), axis=-1)
        self.cached_penc = tf.repeat(
            emb[None, :, :, :org_channels], tf.shape(inputs)[0], axis=0
        )
        #return tf.concat([self.cached_penc, inputs], axis=-1)
        return self.cached_penc + inputs

class PositionalEmbedding(layers.Layer):
  # https://www.tensorflow.org/text/tutorials/transformer
  def __init__(self, vocab_size=21, d_model=16):
    super().__init__()
    self.d_model = d_model
    self.vocab_size = vocab_size
    self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=d_model, mask_zero=True) 
    self.pos_encoding = positional_encoding(length=256, depth=d_model)

  def compute_mask(self, *args, **kwargs):
    return self.embedding.compute_mask(*args, **kwargs)

  def get_config(self):
      config = super(FeedForward, self).get_config()
      config.update({
         "d_model": self.d_model,
         "vocab_size": self.vocab_size
         })
      return config

  def call(self, x):

    length = tf.shape(x)[1]
    x = self.embedding(x)
    # This factor sets the relative scale of the embedding and positonal_encoding.
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    temp = self.pos_encoding[tf.newaxis, :length, :]
    x = x + temp
    return x

class GlobalSelfAttention(layers.Layer):
  def __init__(self, **kwargs):
    # https://www.tensorflow.org/text/tutorials/transformer
    super().__init__()
    self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
    self.layernorm = layers.LayerNormalization()
    self.add = tf.keras.layers.Add()


  def get_config(self):
      config = super(FeedForward, self).get_config()
      config.update(self.mha.get_config())
      return config

  def call(self, x):
    attn_output = self.mha(
        query=x,
        value=x,
        key=x)
    x = self.add([x, attn_output])
    x = self.layernorm(x)
    return x
  
class FeedForward(layers.Layer):
  def __init__(self, d_model, dff, dropout_rate=0.1):
    # https://www.tensorflow.org/text/tutorials/transformer
    super().__init__()
    
    self.seq = tf.keras.Sequential([
      tf.keras.layers.Dense(dff),
      keras.layers.ReLU(negative_slope=0.1),
      tf.keras.layers.Dense(d_model),
      keras.layers.ReLU(negative_slope=0.1),
    ])
    self.dropout = layers.Dropout(dropout_rate)
    self.add = tf.keras.layers.Add()
    self.layer_norm = layers.LayerNormalization()

    self.d_model = d_model
    self.dff = dff

  def get_config(self):
      config = super(FeedForward, self).get_config()
      config.update({
         "d_model": self.d_model,
         "dff": self.dff
         })
      return config

  def call(self, x):
    x = self.add([x, self.seq(x)])
    x = self.dropout(x)
    x = self.layer_norm(x) 
    return x

class EncoderLayer(layers.Layer):
  def __init__(self,*, d_model, num_heads, dff, dropout_rate=0.1):
    super().__init__()

    self.self_attention = GlobalSelfAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate)

    self.ffn = FeedForward(d_model, dff)

    self.d_model = d_model
    self.num_heads = num_heads
    self.dff = dff

  def get_config(self):
      config = super(EncoderLayer, self).get_config()
      config.update({
         "d_model": self.d_model,
         'num_heads': self.num_heads,
         "dff": self.dff
         })
      return config

  def call(self, x):
    start_x = x
    x = self.self_attention(x)
    x = self.ffn(x)
    return x + start_x

class CrossAttention(layers.Layer):
    def __init__(self, **kwargs):
        # https://www.tensorflow.org/text/tutorials/transformer
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = layers.LayerNormalization()
        self.add = tf.keras.layers.Add()

    def call(self, x, context):
        attn_output = self.mha(
            query=x,
            key=context,
            value=context,
            return_attention_scores=False)

        # # Cache the attention scores for plotting later.
        # self.last_attn_scores = attn_scores

        x = self.add([x, attn_output])
        x = self.layernorm(x)

        return x

class DecoderLayer(layers.Layer):
  
  def __init__(self,
               *,
               d_model,
               num_heads,
               dff,
               dropout_rate=0.1):
    super(DecoderLayer, self).__init__()

    self.self_attention = TriangularAttention(
        num_heads=num_heads,
        d_model=d_model,
        dropout_rate=dropout_rate)

    self.cross_attention = CrossAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate)

    self.ffn = FeedForward(d_model, dff)

  def call(self, x, context):
    x = self.self_attention(x=x)
    x = self.cross_attention(x=x, context=context)

    # Cache the last attention scores for plotting later
    #self.last_attn_scores = self.cross_attention.last_attn_scores

    x = self.ffn(x)  # Shape `(batch_size, seq_len, d_model)`.
    return x
  
class Encoder(layers.Layer):
  def __init__(self, *, num_layers, dff, num_heads,
               d_model=32, vocab_size=21, dropout_rate=0.1):
    super().__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.pos_embedding = PositionalEmbedding(
        vocab_size=vocab_size, d_model=d_model)

    self.num_heads = num_heads
    self.vocab_size = vocab_size
    self.dff = dff

    self.enc_layers = [
        EncoderLayer(d_model=d_model,
                     num_heads=num_heads,
                     dff=dff,
                     dropout_rate=dropout_rate)
        for _ in range(num_layers)]
    self.dropout = tf.keras.layers.Dropout(dropout_rate)

  def get_config(self):
      config = super(Encoder, self).get_config()
      config.update({
         "d_model": self.d_model,
         'num_layers': self.num_layers,
          "num_heads": self.num_heads,
          "vocab_size": self.vocab_size,
          "dff": self.dff,
         })
      return config

  def call(self, x):
    if isinstance(x, dict):
      x = x['primary_onehot']
    x = tf.argmax(x, axis=-1)

    # `x` is token-IDs shape: (batch, seq_len)
    x = self.pos_embedding(x)  # Shape `(batch_size, seq_len, d_model)`.

    # Add dropout.
    x = self.dropout(x)

    for i in range(self.num_layers):
      x = self.enc_layers[i](x)

    return x
  
class Decoder(layers.Layer):
  def __init__(self, *, num_layers, d_model, num_heads, dff,
               dropout_rate=0.1):
    super(Decoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.pos_embedding = TFPositionalEncoding2D(d_model-1)
    self.upsample = layers.UpSampling2D(size=256, interpolation="nearest")

    self.dropout = tf.keras.layers.Dropout(dropout_rate)
    self.dec_layers = [
        DecoderLayer(d_model=d_model, num_heads=num_heads,
                     dff=dff, dropout_rate=dropout_rate)
        for _ in range(num_layers)]

    self.last_attn_scores = None

    self.concatenate = layers.Concatenate()

  def call(self, x, context):
    # `x` is token-IDs shape (batch, target_seq_len)

    x = self.pos_embedding(x)  # (batch_size, target_seq_len, d_model)

    #start_shape = x.shape
    #x = tf.reshape(x,shape=(start_shape[0],start_shape[1]*start_shape[2],-1))

    x = self.dropout(x)

    for i in range(self.num_layers):
      x  = self.dec_layers[i](x, context)

    return x



class Transformer(keras.Model):
  def __init__(self, *, dff=512,
               dropout_rate=0.1):
    super().__init__()
    self.encoder = Encoder(num_layers=3,
                           num_heads=3, dff=dff,
                           dropout_rate=dropout_rate)

    self.decoder = Decoder(num_layers=3, d_model=2,
                           num_heads=2, dff=dff,
                           dropout_rate=dropout_rate)
    self.dropout = layers.Dropout(0.1)

    self.normalizer = layers.Normalization()

    self.conv = layers.Conv2D(1,1)
    self.generator = Generator(embedding_size=2)

  def call(self, context, training=False):
     context = self.encoder(context,training)
     context = self.dropout(context, training=training)
     
     x = self.generator(x,training)
     x = self.decoder(x,context,training)

     x = self.conv(x)
     return x
