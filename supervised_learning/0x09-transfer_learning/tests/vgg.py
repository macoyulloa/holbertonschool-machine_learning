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

    train_generator = K.preprocessing.image.ImageDataGenerator(
        rotation_range=2,
        horizontal_flip=True,
        zoom_range=.1)

    test_generator = K.preprocessing.image.ImageDataGenerator(
        rotation_range=2,
        horizontal_flip=True,
        zoom_range=.1)

    train_generator.fit(x_train)
    test_generator.fit(x_test)

    lrr = K.callbacks.ReduceLROnPlateau(
        monitor='val_acc',
        factor=.01,
        patience=3,
        min_lr=1e-5)

    base_model_1 = K.applications.VGG19(include_top=False,
                                        weights='imagenet',
                                        input_shape=(32, 32, 3),
                                        classes=y_train.shape[1])

    model_1 = K.Sequential()
    model_1.add(base_model_1)
    model_1.add(K.layers.Flatten())
    model_1.add(K.layers.Dense(1024, activation=('relu'), input_dim=512))
    model_1.add(K.layers.Dense(512, activation=('relu')))
    model_1.add(K.layers.Dense(256, activation=('relu')))
    model_1.add(K.layers.Dropout(.3))
    model_1.add(K.layers.Dense(128, activation=('relu')))
    model_1.add(K.layers.Dropout(.2))
    model_1.add(K.layers.Dense(10, activation=('softmax')))

    model_1.summary()

    batch = 100
    epochs = 1
    learn_rate = .001

    sgd = K.optimizers.SGD(lr=learn_rate, momentum=.9, nesterov=False)
    adam = K.optimizers.Adam(lr=learn_rate, beta_1=0.9,
                             beta_2=0.999, epsilon=None,
                             decay=0.0, amsgrad=False)

    model_1.compile(optimizer=sgd,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

    history = model_1.fit_generator(train_generator.flow(x_train,
                                                         y_train,
                                                         batch_size=batch),
                                    epochs=epochs,
                                    steps_per_epoch=x_train.shape[0]//batch,
                                    callbacks=[lrr],
                                    verbose=1,
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
