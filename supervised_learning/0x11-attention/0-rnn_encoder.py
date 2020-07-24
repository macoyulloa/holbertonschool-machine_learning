#!/usr/bin/env python3
""" Machine Translation model with RNN's """

import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """ RNN Encoder part of the translation model
    """

    def __init__(self, vocab, embedding, units, batch):
        """ initialized the variables

        Arg:
        - vocab: int the size of the input vocabulary
        - embedding: int dimensionality of the embedding vector
        - units: int the number of hidden units in the RNN cell
        - batch: int representing the batch size

        Public instance attributes:

        - batch: the batch size
        - units: the number of hidden units in the RNN cell
        - embedding: a keras Embedding layer converts words from the
                    vocabulary into an embedding vector
        - gru: a keras GRU layer with units units
        """
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(
            vocab, embedding)
        self.gru = tf.keras.layers.GRU(
            self.units,
            recurrent_initializer="glorot_uniform",
            return_sequences=True,
            return_state=True)

    def initialize_hidden_state(self):
        """ Initializes the hidden states for the RNN cell to
            a tensor of zeros
        """
        return tf.zeros(
            (self.batch, self.units), dtype=tf.dtypes.float32)

    def call(self, x, initial):
        """ Calling the GRU RNN model to construct the encoding part of the
            tranlation model

        Arg:
        - x: tensor of shape (batch, input_seq_len) containing the
            input to the encoder layer as word indices within the vocab
        - initial: tensor of shape (batch, units) with the initial hidden state

        Return:
        - full_seq_outputs: tensor of shape (batch, input_seq_len, units) with
                            the outputs of the encoder
        - lat_hidden_state: tensor of shape (batch, units) containing the last
                            hidden state of the encoder
        """
        embedding = self.embedding(x)
        full_seq_outputs, last_hidden_state = self.gru(embedding,
                                                       initial_state=initial)
        return full_seq_outputs, last_hidden_state
