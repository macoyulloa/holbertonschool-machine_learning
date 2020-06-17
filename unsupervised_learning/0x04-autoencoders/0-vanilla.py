#!/usr/bin/env python3
"""vanilla autoencoder"""
import tensorflow.keras as K


def autoencoder(input_dims, hidden_layers, latent_dims):
    """that creates an autoencoder:

    Arg:
        - input_dims: is an integer containing the dims of the model input
        - hidden_layers: is a list containing the number of nodes for each
                        hidden layer in the encoder, respectively
        - latent_dims: is an integer containing the dimensions of the latent
                    space representation

    Returns: encoder, decoder, auto
        - encoder: is the encoder model
        - decoder: is the decoder model
        - auto: is the full autoencoder model
    """
    # creating the vanila autoencoder model
    input_img = K.layers.Input(shape=(input_dims,))
    # encoded part of the model
    for i, layer in enumerate(hidden_layers):
        if i == 0:
            encoded = K.layers.Dense(layer, activation='relu')(input_img)
        else:
            encoded = K.layers.Dense(layer, activation='relu')(encoded)
    # the botneckle layer
    botneckle = K.layers.Dense(latent_dims, activation='relu')(encoded)
    # decoded part of the model
    for i in range(len(hidden_layers)-1, -1, -1):
        if i == len(hidden_layers)-1:
            decoded = K.layers.Dense(
                hidden_layers[i], activation='relu')(botneckle)
        else:
            decoded = K.layers.Dense(
                hidden_layers[i], activation='relu')(decoded)
    decoded = K.layers.Dense(input_dims, activation='sigmoid')(decoded)
    # mapping the complete autoencoded model, reconstruc the image
    autoencoder = K.models.Model(input_img, decoded)

    # encoder: compressing the input until the botneckle, encoded representation
    encoder = K.models.Model(input_img, botneckle)

    # decoder: mappin the input to reconstruct and decoder the input.
    # input of the moddel decorder
    latent_layer = K.layers.Input(shape=(latent_dims,))
    # output of the model decoder
    for i, layer in enumerate(autoencoder.layers[len(hidden_layers)+2:]):
        if i == 0:
            d_layer = layer(latent_layer)
        else:
            d_layer = layer(d_layer)

    decoder = K.models.Model(latent_layer, d_layer)

    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    return (encoder, decoder, autoencoder)
