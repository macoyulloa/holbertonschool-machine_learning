#!/usr/bin/env python3
"""building a deep learning model using keras"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1,
                decay_rate=1, save_best=False, filepath=None,
                verbose=True, shuffle=False):
    """
    Arg:
        network: is the model to optimize
        data: numpy (m, nx)
        labels: one-hot code shape (m, classes)
        batch_size: size of the batch used foor mini-batch
        epochs: number of passes for data
        validation_data: data to validate the model
        early_stopping: booblean indicates where it should be
        patience: for early stopping
        learning_rate_decay: indicates if it should be used
        alpha: initial lr
        decay_rate: decay rate
        save_best: if sve the model after each epoch
        filepath: where the model should be saved
        verbose: determines if output should be printed
        shuffle: determines whether to shuffle
    return: History object generated
    """
    if validation_data:
        es = K.callbacks.EarlyStopping(monitor='val_loss',
                                       patience=patience)
    else:
        es = None

    if save_best:
        save = K.callbacks.ModelCheckpoint(filepath)
    else:
        save = None

    history = network.fit(x=data, y=labels, epochs=epochs,
                          batch_size=batch_size,
                          validation_data=validation_data,
                          verbose=verbose, shuffle=shuffle,
                          callbacks=[es, save])
    return history
