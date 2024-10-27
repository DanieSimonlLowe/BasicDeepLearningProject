import tensorflow as tf
from keras import layers

class TriangularAttention(layers.Layer):
    def __init__(self, num_heads, d_model, dropout_rate=0.1):
        super(TriangularAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.depth = d_model // num_heads
        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)
        
        self.dense = layers.Dense(d_model)
    
    def split_heads(self, x, batch_size):
        # Split the last dimension into (num_heads, depth).
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])  # (batch_size, num_heads, seq_len, depth)

    def scaled_dot_product_attention(self, q, k, v, mask):
        """Calculate the attention weights."""
        matmul_qk = tf.matmul(q, k, transpose_b=True)  # (batch_size, num_heads, seq_len_q, seq_len_k)

        # Scale the logits by the square root of the depth
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)  # Add large negative number to masked positions

        # Softmax to get the weights
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (batch_size, num_heads, seq_len_q, seq_len_k)

        # Weighted sum of values
        output = tf.matmul(attention_weights, v)  # (batch_size, num_heads, seq_len_q, depth_v)

        return output, attention_weights

    def call(self, pair_representation, mask=None):
        """
        pair_representation: (batch_size, seq_len, seq_len, d_model)
        """
        batch_size = tf.shape(pair_representation)[0]

        # Linear layers for q, k, v
        q = self.wq(pair_representation)  # (batch_size, seq_len, seq_len, d_model)
        k = self.wk(pair_representation)
        v = self.wv(pair_representation)

        # Split heads
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len, depth)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # Scaled dot-product attention
        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)

        # Concatenate heads
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len, num_heads, depth)
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))  # (batch_size, seq_len, d_model)

        # Final linear layer
        output = self.dense(concat_attention)  # (batch_size, seq_len, d_model)
        
        return output

