#!/usr/bin/env python3
"""Transfer learning"""

import tensorflow.keras as K
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':

    (x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()

    x_train = K.applications.resnet50.preprocess_input(x_train)
    x_test = K.applications.resnet50.preprocess_input(x_test)

    y_train = K.utils.to_categorical(y_train, 10)
    y_test = K.utils.to_categorical(y_test, 10)

    expand_x_train = np.fliplr(x_train)
    expand_y_train = np.fliplr(y_train)
    #expand the samples with numpy
    print(expand_x_train.shape)
    print(expand_y_train.shape)
    plt.imshow(expand_x_train[0,:,:,0])
    plt.show()
    plt.imshow(x_train[0,:,:,0])
    plt.show()
    x_train = np.concatenate([x_train, expand_x_train])
    y_train = np.concatenate([y_train, expand_y_train])
    print(x_train.shape)
    print(y_train.shape)

    base_model_2 = K.applications.ResNet50(include_top=False,
                                           weights='imagenet',
                                           input_shape=(32, 32, 3),
                                           classes=y_train.shape[1])

    for layer in base_model_2.layers:
        layer.trainable = False
    #freeze the layers of the model trained

    model_1 = K.Sequential()
    model_1.add(base_model_2)
    model_1.add(K.layers.Flatten())
    model_1.add(K.layers.Dense(10, activation=('softmax')))

    model_1.summary()

    batch = 64
    epochs = 40
    lrate = .001

    sgd = K.optimizers.SGD(lr=lrate, momentum=0.9, nesterov=False)
    adam = K.optimizers.Adam(lr=lrate,
                             beta_1=0.9,
                             beta_2=0.999)

    model_1.compile(optimizer=adam,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

    save = K.callbacks.ModelCheckpoint('cifar10.h5',
                                       monitor='val_acc',
                                       save_best_only=True)

    es = K.callbacks.EarlyStopping(monitor='val_loss',
                                   patience=3)

    lrr = K.callbacks.ReduceLROnPlateau(
        monitor='val_acc',
        factor=.01,
        patience=3,
        min_lr=1e-5)

    history = model_1.fit(x_train,
                          y_train,
                          epochs=epochs,
                          batch_size=batch,
                          verbose=1,
                          callbacks=[lrr, save, es],
                          validation_data=(x_test, y_test))

    model_1.save('cifar10.h5')


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
