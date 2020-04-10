#!/usr/bin/env python3
"""Transfer learning"""

import tensorflow.keras as K


if __name__ == '__main__':

    (x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()

    x_train = K.applications.densenet.preprocess_input(x_train)
    x_test = K.applications.densenet.preprocess_input(x_test)

    y_train = K.utils.to_categorical(y_train, 10)
    y_test = K.utils.to_categorical(y_test, 10)

    print(x_train.shape)
    print(x_test.shape)

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

    new_input = K.Input(shape=(32, 32, 3))
    base_model = K.applications.densenet.DenseNet121(include_top=False,
                                                     input_shape=(32, 32, 3),
                                                     weights='imagenet',
                                                     input_tensor=new_input,
                                                     classes=y_train.shape[1])

    model = K.models.Sequential()
    model.add(base_model)
    model.add(K.layers.GlobalAveragePooling2D())
    model.add(K.layers.Dense(10, activation='softmax'))

    model.summary()

    batch = 128
    epochs = 10
    learn_rate = .001

    sgd = K.optimizers.SGD(lr=learn_rate, momentum=.9, nesterov=False)
    adam = K.optimizers.Adam(lr=learn_rate, beta_1=0.9,
                             beta_2=0.999, epsilon=None,
                             decay=0.0, amsgrad=False)

    model.compile(optimizer=sgd,
                  loss='categorical_crossentropy',
                  metrics=['acc'])

    checkpoint = K.callbacks.ModelCheckpoint('cifar10.h5',
                                             monitor='val_acc',
                                             save_best_only=True)

    lrr = K.callbacks.ReduceLROnPlateau(monitor='val_acc',
                                        factor=.01,
                                        patience=3,
                                        min_lr=1e-5)

    history = model.fit_generator(train_generator.flow(x_train,
                                                       y_train,
                                                       batch_size=batch),
                                  epochs=epochs,
                                  steps_per_epoch=x_train.shape[0]//batch,
                                  callbacks=[checkpoint, lrr],
                                  verbose=1,
                                  validation_data=(x_test, y_test))

    model.save('cifar50.h5')


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
