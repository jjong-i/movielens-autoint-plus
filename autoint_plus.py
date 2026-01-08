import tensorflow as tf
from tensorflow.keras import layers

class AutoIntPlusTF(tf.keras.Model):
    def __init__(self, field_dims, embed_dim=16, head_num=2, attn_layers=3, mlp_dims=(64, 32), dropout=0.2):
        super(AutoIntPlusTF, self).__init__()
        self.embeddings = [layers.Embedding(dim, embed_dim) for dim in field_dims]
        self.attn_blocks = [
            layers.MultiHeadAttention(num_heads=head_num, key_dim=embed_dim, dropout=dropout)
            for _ in range(attn_layers)
        ]
        self.mlp = tf.keras.Sequential()
        for dim in mlp_dims:
            self.mlp.add(layers.Dense(dim, activation='relu'))
            self.mlp.add(layers.Dropout(dropout))
        self.final_linear = layers.Dense(1)

    def call(self, inputs, training=False):
        embeds = [self.embeddings[i](inputs[:, i]) for i in range(inputs.shape[1])]
        embeds = tf.stack(embeds, axis=1) 
        
        # Path 1: Attention
        attn_out = embeds
        for block in self.attn_blocks:
            attn_out = block(attn_out, attn_out, training=training)
        attn_out = layers.Flatten()(attn_out)
        
        # Path 2: MLP
        mlp_out = self.mlp(layers.Flatten()(embeds), training=training)
        
        combined = tf.concat([attn_out, mlp_out], axis=1)
        return tf.squeeze(self.final_linear(combined), axis=1)
