#!/usr/bin/env python3
"""Transfer learning"""

import tensorflow.keras as K
import numpy as np


if __name__ == '__main__':

    """ transfer learning of the model densenet121
        and save it in a file cifar10.h5
    """
    (x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()
    x_train = K.applications.densenet.preprocess_input(x_train)
    y_train = K.utils.to_categorical(y_train, 10)
    x_test = K.applications.densenet.preprocess_input(x_test)
    y_test = K.utils.to_categorical(y_test, 10)

    expand_x_train = np.fliplr(x_train)
    print(expand_x_train.shape)

    x_train = np.concatenate([x_train, expand_x_train])
    y_train = np.concatenate([y_train, y_train])
    print(x_train.shape)
    print(y_train.shape)

    base_model = K.applications.densenet.DenseNet121(
        include_top=False,
        pooling='max',
        input_shape=(32, 32, 3),
        weights='imagenet')

    base_model.summary()

    x = base_model.layers[-1].output
    x = K.layers.Dense(128, activation='relu')(x)
    x = K.layers.Dropout(0.3)(x)
    x = K.layers.Dense(10, activation='softmax')(x)
    model = K.models.Model(inputs=base_model.inputs, outputs=x)

    es = K.callbacks.EarlyStopping(monitor='val_acc',
                                   mode='max',
                                   patience=5)

    save = K.callbacks.ModelCheckpoint('cifar10.h5',
                                       monitor='val_acc',
                                       mode='max',
                                       save_best_only=True)

    lrr = K.callbacks.ReduceLROnPlateau(
        monitor='val_acc',
        factor=.01,
        patience=3,
        min_lr=1e-5
    )

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['acc'])

    history = model.fit(x_train, y_train,
                        validation_data=(x_test, y_test),
                        batch_size=128,
                        callbacks=[es, save, lrr],
                        epochs=30,
                        verbose=1)

    model.save('cifar10.h5')


def preprocess_data(X, Y):
    """pre-processes the data model
    Arg:
        X - numpy.ndarray (m, 32, 32, 3) CIFAR 10 data.
        Y - numpy.ndarray (m,) CIFAR 10 labels for X
    Return:
        Returns: X_p, Y_p
        X_p is a numpy.ndarray preprossesed X
        Y_p is a numpy.ndarray preprocessed Y
    Notes: m is the number of data points
    """
    X = K.applications.densenet.preprocess_input(X)
    Y = K.utils.to_categorical(Y, 10)
    return X, Y
