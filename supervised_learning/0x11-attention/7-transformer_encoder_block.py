#!/usr/bin/env python3
""" Transformers Vanilla Model """

import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    """ Perform encoder block transformer
    """

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """ initialized the variables

        Arg:
        - dm: integer representing the dimensionality of the model
        - h: integer representing the number of heads
        - hidden: the number of hidden units in the fully connected layer
        - drop_rate: the dropout rate

        Public instance attributes:
        - mha: a MultiHeadAttention layer
        - dense_hidden: the hidden dense layer with hidden units and
                        relu activation
        - dense_output: the output dense layer with dm units
        - layernorm1: the first layer norm layer, with epsilon=1e-6
        - layernorm2: the second layer norm layer, with epsilon=1e-6
        - dropout1: the first dropout layer
        - dropout2: the second dropout layer
        """
        super(EncoderBlock, self).__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """ Calling the transformers vanilla model to construct the encoded
            part of the transformer tranlation model

        Arg:
        - x: tensor of shape (batch, input_seq_len, dm)containing the input
                to the encoder block
        - training: boolean to determine if the model is training
        - mask: the mask to be applied for multi head attention

        Return:
        - tensor of shape (batch, input_seq_len, dm) with the block’s output
        """
        input = self.dense_hidden(x)
        self.mha()
        if training:
            self.dropout1()
        self.layernorm1()
        self.dense_output()
        if training:
            self.dropout2()
        self.layernorm2()
        return 1
