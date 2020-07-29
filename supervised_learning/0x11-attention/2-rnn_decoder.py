#!/usr/bin/env python3
""" Machine Translation model with RNN's """

import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """ RNN Decoder part of the translation model
    """

    def __init__(self, vocab, embedding, units, batch):
        """ initialized the variables

        Arg:
        - batch: int with the batch size
        - vocab: int the size of the input vocabulary
        - embedding: int dimensionality of the embedding vector
        - units: int the number of hidden units in the RNN cell
        - batch: int representing the batch size

        Public instance attributes:

        - embedding: a keras Embedding layer converts words from the
                    vocabulary into an embedding vector
        - gru: a keras GRU layer with units units
        - F: a Dense layer with vocab units
        """
        super(RNNDecoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(
            vocab, embedding)
        self.gru = tf.keras.layers.GRU(
            units,
            recurrent_initializer="glorot_uniform",
            return_sequences=True,
            return_state=True)
        self.F = tf.keras.layers.Dense(vocab)

    def call(self, x, s_prev, hidden_states):
        """ Calling the GRU RNN layer to construct the dencoding part of the
            tranlation model

        Arg:
        - x: tensor of shape (batch, 1) containing the previous word in the
                target sequence as an index of the target vocabulary
        - s_prev: is a tensor of shape (batch, units) containing the previous
                decoder hidden state
        - hidden_states: is a tensor of shape (batch, input_seq_len, units)
                containing the outputs of the encoder

        Return:
        - y: tensor of shape (batch, vocab) with the output word as a one hot
                vector in the target vocabulary
        - s: tensor of shape (batch, units) with the new decoder hidden state
        """
        # embedding the input x
        embedding = self.embedding(x)
        # self-attention of the inputs per state
        attention = SelfAttention(s_prev.shape[1])
        # get the attention of the inputs
        context, weights = attention(s_prev, hidden_states)
        # getting the context per each input, decoder part
        context = tf.expand_dims(context, axis=1)
        # concat the context plus the embedding to pass thought the model
        inputs = tf.concat([embedding, context], -1)
        decode_outs, state = self.gru(inputs,
                                      initial_state=hidden_states[:, -1])
        # get the outputs of the RNN attention model
        y = tf.reshape((decode_outs), [-1, decode_outs.shape[2]])
        # reduce the output to the vocab len
        y = self.F(y)

        return y, state
