#!/usr/bin/env python3
"""convolutional autoencoder"""
import tensorflow.keras as K


def autoencoder(input_dims, filters, latent_dims):
    """convolutional autoencoder:

    - Each convolution in the encoder: kernel size of (3, 3), same padding
        and relu activation, followed by max pooling of size (2, 2)
    - Each convolution in the decoder: except for the last two, filter size
        of (3, 3), same padding, relu activat, follow by upsampling size (2, 2)
        - The second to last convolution should instead use valid padding
        - The last convolution: 1 filter, sigmoid activation and no upsampling

    Arg:
        - input_dims: tuple of int containing the dimensions of the model input
        - filters: list containing the number of filters for each conv layer in
                    the encoder, respectively. Reversed it for the decoder.
        - latent_dims: is a tuple of integers containing the dimensions of the
                        latent space representation

    Returns: encoder, decoder, auto
        - encoder: is the encoder model
        - decoder: is the decoder model
        - auto: is the full autoencoder model
    """
    # creating the convolutional autoencoder model

    # encoded part of the model
    input_img = K.Input(shape=(input_dims))
    encoded = K.layers.Conv2D(filters=filters[0], kernel_size=3,
                              padding='same',
                              activation='relu')(input_img)
    encoded_pool = K.layers.MaxPooling2D(pool_size=[2, 2])(encoded)

    for i in range(1, len(filters)):
        encoded = K.layers.Conv2D(filters=filters[i], kernel_size=3,
                                  padding='same',
                                  activation='relu')(encoded_pool)
        encoded_pool = K.layers.MaxPooling2D(pool_size=[2, 2])(encoded)
    # the botneckle layer, the latent space
    flatten = K.layers.Flatten()(encoded_pool)
    latent = K.layers.Dense(latent_dims)(flatten)
    # encoder: compressing the input until the botneckle, encoded repre
    encoder = K.models.Model(input_img, latent)

    # decoded part of the model
    input_decoder = K.Input(shape=(latent_dims))
    for i in range(len(filters)-1, -1, -1):
        if i == len(filters)-1:
            decoded = K.layers.Conv2D(filters=filters[i], kernel_size=3,
                                      padding='same',
                                      activation='relu')(input_decoder)
            decoded_pool = K.layers.UpSampling2D(pool_size=[2, 2])(encoded)
        if i == 0:
            decoded = K.layers.Conv2D(filters=filters[i], kernel_size=3,
                                      padding='valid',
                                      activation='relu')(decoded_pool)
            decoded_pool = K.layers.UpSampling2D(pool_size=[2, 2])(decoded)
        else:
            decoded = K.layers.Conv2D(filters=filters[i], kernel_size=3,
                                      padding='same',
                                      activation='relu')(decoded_pool)
            decoded_pool = K.layers.UpSampling2D(pool_size=[2, 2])(decoded)
    decoded = K.layers.Dense(filters=1, kernel_size=3,
                             padding='same',
                             activation='sigmoid')(decoded_pool)
    # decoder: mappin the input to reconstruct and decoder the input.
    decoder = K.models.Model(input_decoder, decoded)

    input_autoencoder = K.Input(shape=(input_dims))
    encoder_outs = encoder(input_autoencoder)
    decoder_outs = decoder(encoder_outs)
    # mapping the complete autoencoded model, reconstruc the image
    autoencoder = K.models.Model(
        inputs=input_autoencoder, outputs=decoder_outs)

    autoencoder.compile(optimizer='Adam', loss='binary_crossentropy')

    return (encoder, decoder, autoencoder)
