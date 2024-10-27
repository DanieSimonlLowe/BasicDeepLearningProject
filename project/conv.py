import keras
from keras import layers
import tensorflow as tf
from transformer import Encoder, CrossAttention, TFPositionalEncoding2D
import math

@keras.saving.register_keras_serializable()
def sinusoidal_embedding(x):
    embedding_min_frequency = 1.0
    embedding_max_frequency = 1000.0
    embedding_dims = 16
    frequencies = tf.exp(
        tf.linspace(
            tf.math.log(embedding_min_frequency),
            tf.math.log(embedding_max_frequency),
            embedding_dims // 2,
        )
    )
    angular_speeds = tf.cast(2.0 * math.pi * frequencies, "float32")
    embeddings = tf.concat(
        [tf.sin(angular_speeds * x), tf.cos(angular_speeds * x)], axis=3
    )
    return embeddings


class ResidualBlock(layers.Layer):
    def __init__(self, width, change_width = False, conv_size=3):
        super().__init__()
        self.width = width
        self.conv1 = layers.Conv2D(width,conv_size,padding='same')
        self.conv2 = layers.Conv2D(width,conv_size,padding='same')
        self.dropout = layers.Dropout(rate=0.1)
        self.add = layers.Add()
        self.normalization = layers.BatchNormalization(center=False, scale=False)
        self.change_width = change_width
        self.width = width
        self.conv_size = conv_size

        self.layernorm = layers.LayerNormalization()

        if self.change_width:
            self.conv_change = layers.Conv2D(self.width, kernel_size=1)

    def get_config(self):
      config = super(ResidualBlock, self).get_config()
      config.update({
            "conv_size": self.self.conv_size,
            'width': self.width,
            'change_width': self.change_width,
            })
      return config

    def call(self, x):
        input_width = x.shape[3]
        if input_width != self.width:
            #residual = layers.Conv2D(self.width, kernel_size=1)(x)
            if self.change_width:
                x = self.conv_change(x)
            else:
                raise RuntimeError('wrong width')
        residual = x

        x = self.normalization(x)
        x = self.conv1(x)
        x = keras.activations.relu(x,alpha=0.1)
        x = self.conv2(x)
        x = keras.activations.relu(x,alpha=0.1)
        
        x = self.add([x, residual])

        x = self.layernorm(x)
        x = self.dropout(x)

        return x

class DownBlock(layers.Layer):
    def __init__(self, block_depth, width, heads=None, key_dim=None, value_dim=None, has_cross=False):
        super().__init__()
        self.blocks = [ResidualBlock(width,heads,key_dim,value_dim,change_width=True, has_cross=has_cross)] + \
            [ResidualBlock(width,heads,key_dim,value_dim,has_cross=has_cross) for _ in range(block_depth-1)]
        self.pool = layers.AveragePooling2D(pool_size=2)
    
    def call(self, x, context):
        x, skips = x
        for block in self.blocks:
            x = block(x, context)
            skips.append(x)
        x = self.pool(x)
        return x, skips
    
class UpBlock(layers.Layer):
    
    def __init__(self, block_depth, width, heads, key_dim, value_dim, has_cross=False, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)

        self.blocks = [ResidualBlock(width,heads,key_dim,value_dim,change_width=True, has_cross=has_cross)] + \
            [ResidualBlock(width,heads,key_dim,value_dim,has_cross=has_cross) for _ in range(block_depth-1)]
        self.pool = layers.UpSampling2D(size=(2,2), interpolation="nearest")
        self.add = layers.Add()
    
    def call(self, x, context):
        x, skips = x
        x = self.pool(x)
        for block in self.blocks:
            x = block(x, context)
            x = self.add([x,skips.pop()])
        return x, skips

class UNetWithAttention(layers.Layer):
    def __init__(self, widths, has_crosses, block_depth=2, image_size=256, heads=3, key_dim=32, value_dim=32):
        super().__init__()
        
        self.embedding = layers.Lambda(sinusoidal_embedding, output_shape=(1, 1, 32))

        self.convInput = layers.Conv2D(widths[0], kernel_size=1)
        self.convOutput = layers.Conv2D(1, kernel_size=1)
        self.concatenate = layers.Concatenate()

        self.upsample = layers.UpSampling2D(size=image_size, interpolation="nearest")

        self.image_size = image_size

        self.downs = [DownBlock(block_depth,widths[i],heads,key_dim,value_dim, has_cross=has_crosses[i]) for i in range(len(widths)-1)]
        self.middle = [ResidualBlock(widths[-1],heads,key_dim,value_dim,has_cross=has_crosses[-1],change_width=True)] + \
            [ResidualBlock(widths[-1],heads,key_dim,value_dim,has_cross=has_crosses[-1]) for _ in range(block_depth-1)]
        self.ups = [UpBlock(block_depth,widths[i],heads,key_dim,value_dim, has_cross=has_crosses[i])  for i in range(len(widths)-2,-1,-1)]


        self.encoder = Encoder(num_layers=4,dff=512,num_heads=3,d_model=widths[-1]+8)

    def call(self, noisy_images, noise_variances, context):

        e = self.embedding(noise_variances)
        e = self.upsample(e)

        x = self.convInput(noisy_images)
        x = self.concatenate([x,e])

        skips = []

        context = self.encoder(context)

        for down in self.downs:
            x, skips = down([x,skips], context)
        
        for res in self.middle:
            x = res(x, context)

        for up in self.ups:
            x, skips = up([x,skips], context)
        
        x = self.convOutput(x)
        return x
     