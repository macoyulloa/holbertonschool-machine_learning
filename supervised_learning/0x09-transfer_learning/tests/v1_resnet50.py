#!/usr/bin/env python3
"""Transfer learning"""

import tensorflow.keras as K


if __name__ == '__main__':
    base_model = K.applications.resnet50.ResNet50(weights='imagenet',
                                                  include_top=False,
                                                  input_shape=(32, 32, 3))

    (x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()

    x_train = K.applications.resnet50.preprocess_input(x_train)
    x_test = K.applications.resnet50.preprocess_input(x_test)

    y_train = K.utils.to_categorical(y_train, 10)
    y_test = K.utils.to_categorical(y_test, 10)

    print(x_train.shape)
    print(x_test.shape)

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.layers[-1].output
    x = K.layers.Flatten()(x)
    x = K.layers.BatchNormalization()(x)
    x = K.layers.Dense(128, activation='relu')(x)
    x = K.layers.Dropout(0.3)(x)
    x = K.layers.BatchNormalization()(x)
    x = K.layers.Dense(64, activation='relu')(x)
    x = K.layers.Dropout(0.3)(x)
    x = K.layers.BatchNormalization()(x)
    predict = K.layers.Dense(10, activation='softmax')(x)

    model = K.models.Model(inputs=base_model.inputs, outputs=predict)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['acc'])

    model.summary()

    checkpoint = K.callbacks.ModelCheckpoint('cifar10.h5',
                                             monitor='val_acc',
                                             save_best_only=True)

    history = model.fit(x_train,
                        y_train,
                        epochs=20,
                        batch_size=800,
                        verbose=1,
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
    X = K.applications.resnet50.preprocess_input(X)
    Y = K.utils.to_categorical(Y, 10)
    return X, Y
