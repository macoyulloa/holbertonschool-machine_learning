#!/usr/bin/env python3
""" Transformers Vanilla Model """

import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """ Perform multi head attention
    """

    def __init__(self, dm, h):
        """ initialized the variables

        Arg:
        - dm: integer representing the dimensionality of the model
        - h: integer representing the number of heads
        - dm: is divisible by h

        Public instance attributes:

        - h: the number of heads
        - dm: the dimensionality of the model
        - depth: the depth of each attention head
        - Wq: Dense layer with dm units, for generate the query matrix
        - Wk: Dense layer with dm units, for generate the key matrix
        - Wv: Dense layer with dm units, for generate the value matrix
        - linear: Dense layer with dm units, for generate the attention output
        """
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.dm = dm
        self.depth = int(self.dm / self.h)
        self.Wq = tf.keras.layers.Dense(self.dm)
        self.Wk = tf.keras.layers.Dense(self.dm)
        self.Wv = tf.keras.layers.Dense(self.dm)
        self.linear = tf.keras.layers.Dense(self.dm)

    def call(self, Q, K, V, mask):
        """ Calling the transformers vanilla model to construct the multi
            head attention tranlation model

        Arg:
        - Q: tensor of shape (batch, seq_len_q, dk) containing the input to
                generate the query matrix
        - K: tensor of shape (batch, seq_len_v, dk) containing the input to
                generate the key matrix
        - V: tensor of shape (batch, seq_len_v, dv) containing the input to
                generate the value matrix
        - mask: is always None

        Return: output, weights
        - output: tensor with its last two dimensions as (..., seq_len_q, dm)
            containing the scaled dot product attention
        - weights: a tensor with its last three dimensions as
            (..., h, seq_len_q, seq_len_v) containing the attention weights
        """
        v_list, k_list, q_list = [], [], []
        V_linear = self.Wv(V)
        K_linear = self.Wk(K)
        Q_linear = self.Wq(Q)

        v_list = tf.split(V_linear, self.h, axis=-1)
        k_list = tf.split(K_linear, self.h, axis=-1)
        q_list = tf.split(Q_linear, self.h, axis=-1)

        output, weight = sdp_attention(
            q_list[0], k_list[0], v_list[0], mask)
        weight = tf.expand_dims(weight, 1)

        for i in range(1, self.h):
            output_s, weight_s = sdp_attention(
                q_list[i], k_list[i], v_list[i], mask)
            weight_s = tf.expand_dims(weight_s, 1)
            output = tf.concat([output, output_s], axis=-1)
            weight = tf.concat([weight, weight_s], axis=1)

        outputs = self.linear(output)
        weights = tf.nn.softmax(weight)

        return outputs, weights
