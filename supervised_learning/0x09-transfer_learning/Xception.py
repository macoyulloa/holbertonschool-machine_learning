#!/usr/bin/env python3
"""Transfer learning"""

import tensorflow.keras as K


if __name__ == '__main__':

    (x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()

    x_train = K.applications.resnet.preprocess_input(x_train)
    x_test = K.applications.resnet.preprocess_input(x_test)

    y_train = K.utils.to_categorical(y_train, 10)
    y_test = K.utils.to_categorical(y_test, 10)

    print(x_train.shape)
    print(x_test.shape)

    base_model = K.applications.xception.Xception(include_top=False,
                                                  input_shape=(128, 128, 3),
                                                  weights='imagenet')

    # Top Model Block
    model = K.models.Sequential()
    model.add(K.layers.UpSampling2D((2, 2)))
    model.add(K.layers.UpSampling2D((2, 2)))
    model.add(base_model)
    model.add(K.layers.GlobalAveragePooling2D())
    model.add(K.layers.Dense(10, activation='softmax'))

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all layers of the based model that is already pre-trained.
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['acc'])

    model.summary()

    checkpoint = K.callbacks.ModelCheckpoint('cifar10.h5',
                                             monitor='val_acc',
                                             save_best_only=True)

    history = model.fit(x_train,
                        y_train,
                        epochs=15,
                        batch_size=256,
                        callbacks=[checkpoint],
                        validation_data=(x_test, y_test))

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
    X = K.applications.xception.preprocess_input(X)
    Y = K.utils.to_categorical(Y, 10)
    return X, Y
