#!/usr/bin/env python3
""" Machine Translation model with RNN's """

import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """ SelfAttention part of the translation model
    """

    def __init__(self, units):
        """ initialized the variables

        Arg:
        - units: int the number of hidden units in the RNN cell

        Public instance attributes:
        - W: Dense layer with units units, to be applied to the
                previous decoder hidden state
        - U: Dense layer with units units, to be applied to the
                encoder hidden states
        - V: Dense layer with 1 units, to be applied to the tanh
                of the sum of the outputs of W and U
        """
        super(SelfAttention, self).__init__()
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """ Calling the matrices to construct the self attention part of the
            tranlation model

        Arg:
        - s_prev: is a tensor of shape (batch, units) containing the previous
                decoder hidden state
        - hidden_states: is a tensor of shape (batch, input_seq_len, units)
                containing the outputs of the encoder

        Return: (context, weights)
        - context: tensor of shape (batch, units) that contains the context
                    vector for the decoder
        - weights: tensor of shape (batch, input_seq_len, 1) that contains the
                    attention weights
        """
        w = self.W(s_prev)
        u = self.U(hidden_states)
        w = tf.expand_dims(w, axis=1)
        e = self.V((tf.nn.tanh(w + u)))
        attention = tf.nn.softmax(e, axis=1)
        c = tf.reduce_sum((attention * hidden_states), axis=1)

        return c, attention
