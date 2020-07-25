#!/usr/bin/env python3
""" Transformers Vanilla Model """

import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """calculates the scaled dot product attention:

    Arg:
    - Q: is a tensor with its last two dimensions as (..., seq_len_q, dk)
        containing the query matrix
    - K: is a tensor with its last two dimensions as (..., seq_len_v, dk)
        containing the key matrix
    - V: is a tensor with its last two dimensions as (..., seq_len_v, dv)
        containing the value matrix
    - mask: tensor that can be broadcast into (..., seq_len_q, seq_len_v)
        containing the optional mask, or defaulted to None

    Returns: output, weights
    - outputs: tensor, the last two dimens as (..., seq_len_q, dv)
            containing the scaled dot product attention
    - weights: tensor, the last two dimens as (..., seq_len_q, seq_len_v)
            containing the attention weights
    """
    q_k_dot_prod = tf.linalg.matmul(Q, K, transpose_b=True)
    dk_square = tf.cast((tf.math.square(K.shape[-1])), tf.float32)
    q_k_scaled = tf.math.divide(q_k_dot_prod, dk_square)
    if mask is not None:
        mask_multiply = tf.math.multiply(mask, -1e9)
        q_k_scaled = tf.math.add(q_k_scaled, mask_multiply)
    q_k_activated = tf.nn.softmax(q_k_scaled)
    q_k_v = tf.linalg.matmul(q_k_activated, V)

    return q_k_v, q_k_activated
