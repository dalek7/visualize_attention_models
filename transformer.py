import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model

# Written in Intelligent Robotics Lab., Hallym Univ, 2020
# Written in MLSLab, SKKU 2021
#
# Seung-Chan Kim
# Note that this is not the final version
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'projection_dim': self.projection_dim,
            'query_dense': self.query_dense,
            'key_dense': self.key_dense,
            'value_dense': self.value_dense,
            'combine_heads': self.combine_heads
        })
        return config

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)

        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs, return_weights=False):
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)
        query = self.separate_heads(
            query, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(
            key, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(
            value, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(
            attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )  # (batch_size, seq_len, embed_dim)
        output = self.combine_heads(
            concat_attention
        )  # (batch_size, seq_len, embed_dim)

        if return_weights:
            return output, weights
        else:
            return output


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim), ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        #self._name = 'TransformerBlock_{}_{}_{}'.format(embed_dim, num_heads, ff_dim)
        # The name "TransformerBlock_64_4_64" is used 4 times in the model. All layer names should be unique.

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'att': self.att,
            'ffn': self.ffn,
            'layernorm1': self.layernorm1,
            'layernorm2': self.layernorm2,
            'dropout1': self.dropout1,
            'dropout2': self.dropout2,
        })
        return config

    def call(self, inputs, training, return_weights=False):
        if return_weights:
            attn_output, attn_weights = self.att(inputs, return_weights=return_weights)
        else:
            attn_output = self.att(inputs)

        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)

        if return_weights:
            return self.layernorm2(out1 + ffn_output), attn_weights
        else:
            return self.layernorm2(out1 + ffn_output)

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = 100#tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

class TokenAndPositionEmbedding3(layers.Layer): #2023/2/10 not working. need to configure properly.
    def __init__(self, maxlen, data_dim, embed_dim):  # vocab_size, , maxlen=100
        super(TokenAndPositionEmbedding3, self).__init__()
        self.token_emb = layers.Embedding(input_dim=data_dim, output_dim=embed_dim)
        #self.token_emb = layers.Dense(embed_dim, activation="relu")
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)
        self.maxlen = maxlen
        self.data_dim = data_dim
        # self._name = 'TokenAndPositionEmbedding2_L{}_D{}'.format(maxlen, embed_dim) --> check 20210928

    # 20201214 added for saving model
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'token_emb': self.token_emb,
            'pos_emb': self.pos_emb
        })
        return config

    def call(self, x):
        maxlen = self.maxlen  # tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)  # (100,) --> (100, 64) Unique 한 vector로
        x = self.token_emb(x)
        return x + positions  ##
    
    
    
class TokenAndPositionEmbedding2(layers.Layer):
    def __init__(self, maxlen, embed_dim):  # vocab_size, , maxlen=100
        super(TokenAndPositionEmbedding2, self).__init__()
        # self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.token_emb = layers.Dense(embed_dim, activation="tanh") #relu
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)
        self.maxlen = maxlen
        # self._name = 'TokenAndPositionEmbedding2_L{}_D{}'.format(maxlen, embed_dim) --> check 20210928

    # 20201214 added for saving model
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'token_emb': self.token_emb,
            'pos_emb': self.pos_emb
        })
        return config

    def call(self, x):
        maxlen = self.maxlen  # tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)  # (100,) --> (100, 64) Unique 한 vector로
        x = self.token_emb(x)
        return x + positions  ##
    
def make_transformer(maxlen, data_dim, embed_dim, num_heads, ff_dim, n_classes, dropout=0.3, num_transformer_blocks=4, return_logit_only=False):
    #maxlen =  200  # Only consider the first 200 words of each movie review
    # data_dim = 6
    #inputs = layers.Input(shape=(maxlen,))
    
    inputs = layers.Input(shape=(maxlen, data_dim )) 
    if False:
        embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
        x = embedding_layer(inputs)
    else:

        embedding_layer = TokenAndPositionEmbedding2(maxlen, embed_dim) #100,  32 (<--- was 6)
        x = embedding_layer(inputs)
        #x = layers.Dense(embed_dim, activation="relu")(inputs)


    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x, attn_weights = transformer_block(x,training=True, return_weights = True)#

    for _ in range(num_transformer_blocks-1):
        x = transformer_block(x,training=True, return_weights = False)#
        #x = transformer_block(x,training=True, return_weights = False)
        #x = transformer_block(x,training=True, return_weights = False)

    #x = transformer_block(x)
    #x = transformer_block(x)
    #x = transformer_block(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dropout)(x)
    #x = layers.Dense(20, activation="relu")(x)
    #x = layers.Dropout(0.1)(x)
    
    if return_logit_only:
        outputs = x
    else:
        outputs = layers.Dense(n_classes, activation="softmax")(x)
        
    # transformer_block._name
    model = keras.Model(inputs=inputs, outputs=outputs)
    #model = keras.Model(inputs=inputs, outputs=[outputs, attn_weights])
    return model, attn_weights 

def transformer_encoder(inputs, maxlen, embed_dim, num_heads, ff_dim,dropout, n_classes, return_logit_only=False):
    # Block형태로 사용 버젼 2022/01/01
    
    '''
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder_my(x, maxlen, embed_dim, num_heads, ff_dim, dropout, n_classes, True)
    '''
    embedding_layer = TokenAndPositionEmbedding2(maxlen, embed_dim) #100,  32 (<--- was 6)
    
    x = embedding_layer(inputs)
    #x = layers.Dense(embed_dim, activation="relu")(inputs)

    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x, attn_weights = transformer_block(x,training=True, return_weights = True)#

    x = layers.Dropout(dropout)(x)
    
    if return_logit_only:
        outputs = x
    else:
        outputs = layers.Dense(n_classes, activation="softmax")(x)
    return outputs


def transformer_encoder_(inputs, head_size, num_heads, ff_dim, dropout=0):
    # https://keras.io/examples/timeseries/timeseries_transformer_classification/
    # Conv등 확인이 안되었고, 일단 잘 안됨. 최대 70% acc.
    # Normalization and Attention
    x = inputs
    #x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res