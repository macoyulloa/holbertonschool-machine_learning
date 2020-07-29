#!/usr/bin/env python3
""" Transformers Vanilla Model """

import tensorflow as tf

Encoder = __import__('9-transformer_encoder').Encoder
Decoder = __import__('10-transformer_decoder').Decoder


class Transformer(tf.keras.Model):
    " Perform encoder part of the transformer "

    def __init__(self, N, dm, h, hidden, input_vocab, target_vocab,
                 max_seq_input, max_seq_target, drop_rate=0.1):
        """ initialized the variables of the model

        Arg:
        - N: the number of blocks in the encoder
        - dm: integer representing the dimensionality of the model
        - h: integer representing the number of heads
        - hidden: the number of hidden units in the fully connected layer
        - input_vocab - the size of the input vocabulary
        - target_vocab: the size of the target vocabulary
        - max_seq_input: max sequence len possible for the input
        - max_seq_target: max sequence len possible for the target
        - drop_rate: the dropout rate

        Public instance attributes:
        - encoder: the encoder layer
        - decoder: the decoder layer
        - linear: final Dense layer with target_vocab units
        """
        super(Transformer, self).__init__()
        self.encoder = Encoder(N, dm, h, hidden, input_vocab,
                               max_seq_input, drop_rate)
        self.decoder = Decoder(N, dm, h, hidden, target_vocab,
                               max_seq_target, drop_rate)
        self.linear = tf.keras.layers.Dense(target_vocab)

    def call(self, inputs, target, training, encoder_mask,
             look_ahead_mask, decoder_mask):
        """ Transformers vanilla model

        Arg:
        - inputs: tensor of shape (batch, input_seq_len, dm)
                  containing the inputs
        - target: tensor of shape (batch, target_seq_len, dm)
                  containing the target
        - training: boolean to determine if the model is training
        - encoder_mask: the padding mask to be applied to the encoder
        - look_ahead_mask: the look ahead mask to be applied to the decoder
        - decoder_mask: the padding mask to be applied to the decoder

        Return:
        - tensor of shape (batch, target_seq_len, target_vocab) with
           the transformer output
        """
        encoder_output = self.encoder(inputs, training, encoder_mask)
        out_decoder = self.decoder(target, encoder_output, training,
                                   look_ahead_mask, decoder_mask)
        output = self.linear(out_decoder)

        return output
