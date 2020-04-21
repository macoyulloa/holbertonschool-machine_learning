#!/usr/bin/env python3
"""Transfer learning"""

import tensorflow.keras as K
import numpy as np


if __name__ == '__main__':

    """ transfer learning of the model densenet121
        and save it in a file cifar10.h5
    """
    (x_train, y_train), (x_valid, y_valid) = K.datasets.cifar10.load_data()
    x_train = K.applications.densenet.preprocess_input(x_train)
    y_train = K.utils.to_categorical(y_train, 10)
    x_valid_p = K.applications.densenet.preprocess_input(x_valid)
    y_valid_oh = K.utils.to_categorical(y_valid, 10)

    expand_x_train = np.flip(x_train, axis=2)

    x_train_p = np.concatenate([x_train, expand_x_train], axis=0)
    y_train_oh = np.concatenate([y_train, y_train], axis=0)

    inputs1 = K.layers.Input(shape=(32, 32, 3))

    Y = K.layers.Lambda(lambda image: K.backend.resize_images(
        image, int(299/32), int(299/32), "channels_last"))(inputs1)

    base_model1 = K.applications.xception.Xception(
        include_top=False,
        pooling='avg',
        input_tensor=Y)
    x_t = base_model1.predict(x_train_p)
    x_v = base_model1.predict(x_valid_p)

    base_model1.summary()

    inputs2 =K.layers.Input(shape=(2048,))
    init = K.initializers.he_uniform()
    x = K.layers.Dense(512, activation=None,
                       kernel_initializer=init)(inputs2)
    x = K.layers.BatchNormalization()(x)
    x = K.layers.LeakyReLU()(x)
    x = K.layers.Dropout(0.45)(x)
    x = K.layers.Dense(10, activation='softmax',
                       kernel_initializer=init)(x)

    model_classifier = K.models.Model(inputs=inputs2, outputs=x)

    es = K.callbacks.EarlyStopping(monitor='val_loss',
                                   mode='max',
                                   patience=5)

    save = K.callbacks.ModelCheckpoint('cifar10.h5',
                                       monitor='val_loss',
                                       verbose=1,
                                       mode='max',
                                       save_best_only=True)

    sgd = K.optimizers.SGD(0.005, 0.9,
                           decay=0.0001,
                           nesterov=True)

    model_classifier.compile(optimizer=sgd,
                             loss='categorical_crossentropy',
                             metrics=['acc'])

    model_classifier.fit(x_t, y_train_oh,
                         validation_data=(x_v, y_valid_oh),
                         batch_size=512,
                         callbacks=[es, save],
                         epochs=30,
                         verbose=1)

    del model_classifier

    model = K.models.load_model('cifar10.h5')
    x = base_model1.output
    for layer in model.layers[1:]:
        x = layer(x)
    full_model = K.models.Model(inputs=inputs1, outputs=x)
    sgd = K.optimizers.SGD(0.005, 0.9, decay=0.0001, nesterov=True)
    full_model.compile(optimizer=sgd,
                       loss='categorical_crossentropy',
                       metrics=['acc'])
    full_model.save('cifar10.h5')

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
    Y = K.applications.xception.preprocess_input(Y)
    return X, Y
