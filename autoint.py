import tensorflow as tf
from tensorflow.keras import layers

class AutoInt(tf.keras.Model):
    def __init__(self, field_dims, embed_dim=16, head_num=2, attn_layers=3):
        super(AutoInt, self).__init__()
        # Embedding Layer
        self.embeddings = [layers.Embedding(dim, embed_dim) for dim in field_dims]
        
        # Self-Attention Blocks
        self.attn_blocks = [
            layers.MultiHeadAttention(num_heads=head_num, key_dim=embed_dim)
            for _ in range(attn_layers)
        ]
        
        # Output Layer
        self.final_linear = layers.Dense(1)

    def call(self, inputs):
        # inputs: [batch_size, num_fields]
        embeds = [self.embeddings[i](inputs[:, i]) for i in range(inputs.shape[1])]
        embeds = tf.stack(embeds, axis=1) 
        
        attn_out = embeds
        for block in self.attn_blocks:
            attn_out = block(attn_out, attn_out)
            
        attn_out = layers.Flatten()(attn_out)
        return tf.squeeze(self.final_linear(attn_out), axis=1)
