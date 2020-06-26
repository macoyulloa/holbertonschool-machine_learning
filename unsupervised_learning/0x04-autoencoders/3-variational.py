#!/usr/bin/env python3
"""variational autoencoder"""
import tensorflow.keras as K


def autoencoder(input_dims, hidden_layers, latent_dims):
    """that creates a variational autoencoder:

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
    # creating the variational autoencoder model

    # encoded part of the model
    input_x = K.Input(shape=(input_dims,))
    for i, layer in enumerate(hidden_layers):
        if i == 0:
            encoded = K.layers.Dense(layer, activation='relu')(input_x)
        else:
            encoded = K.layers.Dense(layer, activation='relu')(encoded)
    # the botneckle layer: as result as a distribution prob, 2 arrays
    # the z mean of the prob distribution array and the standar desviation
    z_mean = K.layers.Dense(latent_dims)(encoded)
    z_stand_des = K.layers.Dense(latent_dims)(encoded)

    def sampling(args):
        z_mean, z_stand_des = args
        epsilon = K.backend.random_normal(shape=(latent_dims,),
                                          mean=0.0, stddev=1.0)
        return z_mean + K.backend.exp(z_stand_des) * epsilon

    # sampling the data from the data set using the z_mean and z_stand_dev
    z = K.layers.Lambda(sampling, output_shape=(
        latent_dims,))([z_mean, z_stand_des])
    # encoder part of the model take and image and get a sample based
    # on themean, standard desv
    encoder = K.models.Model(input_x, z)
    encoder.summary()

    # decoded part of the model
    input_z = K.Input(shape=(latent_dims,))
    for i in range(len(hidden_layers)-1, -1, -1):
        if i == len(hidden_layers)-1:
            decoded = K.layers.Dense(
                hidden_layers[i], activation='relu')(input_z)
        else:
            decoded = K.layers.Dense(
                hidden_layers[i], activation='relu')(decoded)
    decoded = K.layers.Dense(input_dims, activation='sigmoid')(decoded)
    # decoder: generating an image based in the dataset
    decoder = K.models.Model(input_z, decoded)
    decoder.summary()

    x = K.Input(shape=(input_dims,))
    z_encoder = encoder(x)
    x_decoder_mean = decoder(z_encoder)
    # mapping the complete autoencoded model, reconstruc the image
    autoencoder = K.models.Model(
        inputs=x, outputs=x_decoder_mean)
    autoencoder.summary()

    def vae_loss(x, x_decoder_mean):
        x_loss = K.backend.binary_crossentropy(x, x_decoder_mean)
        kl_loss = - 0.5 * K.backend.mean(1 + z_stand_des -
                                         K.backend.square(z_mean) -
                                         K.backend.exp(z_stand_des), axis=-1)
        return x_loss + kl_loss

    autoencoder.compile(optimizer='Adam', loss=vae_loss)

    return (encoder, decoder, autoencoder)
