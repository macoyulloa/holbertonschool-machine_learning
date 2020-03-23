0x07. Convolutional Neural Networks

Tasks
-----

#### 0\. Convolutional Forward Prop

Write a function `def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):` that performs forward propagation over a convolutional layer of a neural network:

-   `A_prev` is a `numpy.ndarray` of shape `(m, h_prev, w_prev, c_prev)` containing the output of the previous layer
-   `W` is a `numpy.ndarray` of shape `(kh, kw, c_prev, c_new)` containing the kernels for the convolution
-   `b` is a `numpy.ndarray` of shape `(1, 1, 1, c_new)` containing the biases applied to the convolution
-   `activation` is an activation function applied to the convolution
-   `padding` is a string that is either `same` or `valid`, indicating the type of padding used
-   `stride` is a tuple of `(sh, sw)` containing the strides for the convolution
-   you may `import numpy as np`
-   Returns: the output of the convolutional layer

#### 1\. Pooling Forward Prop

Write a function `def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):` that performs forward propagation over a pooling layer of a neural network

#### 2\. Convolutional Back Prop

Write a function `def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):` that performs back propagation over a convolutional layer of a neural network

#### 3\. Pooling Back Prop

Write a function `def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):` that performs back propagation over a pooling layer of a neural network:

#### 4\. LeNet-5 (Tensorflow)

Write a function `def lenet5(x, y):` that builds a modified version of the `LeNet-5` architecture using `tensorflow`

-   `x` is a `tf.placeholder` of shape `(m, 28, 28, 1)` containing the input images for the network
-   `y` is a `tf.placeholder` of shape `(m, 10)` containing the one-hot labels for the network
-   The model should consist of the following layers in order:
  -   Convolutional layer with 6 kernels of shape 5x5 with `same` padding
  -   Max pooling layer with kernels of shape 2x2 with 2x2 strides
  -   Convolutional layer with 16 kernels of shape 5x5 with `valid` padding
  -   Max pooling layer with kernels of shape 2x2 with 2x2 strides
  -   Fully connected layer with 120 nodes
  -   Fully connected layer with 84 nodes
  -   Fully connected softmax output layer with 10 nodes
-   All layers requiring initialization should initialize their kernels with the `he_normal` initialization method: `tf.contrib.layers.variance_scaling_initializer()`
-   All hidden layers requiring activation should use the `relu` activation function
-   you may `import tensorflow as tf`
-   Returns:
   -   a tensor for the softmax activated output
   -   a training operation that utilizes `Adam` optimization (with default hyperparameters)
   -   a tensor for the loss of the netowrk
   -   a tensor for the accuracy of the network

#### 5\. LeNet-5 (Keras)

Write a function `def lenet5(X):` that builds a modified version of the `LeNet-5` architecture using `keras`:

-   `X` is a `K.Input` of shape `(m, 28, 28, 1)` containing the input images for the network
-   The model should consist of the following layers in order
  -   Convolutional layer with 6 kernels of shape 5x5 with `same` padding
  -   Max pooling layer with kernels of shape 2x2 with 2x2 strides
  -   Convolutional layer with 16 kernels of shape 5x5 with `valid` padding
  -   Max pooling layer with kernels of shape 2x2 with 2x2 strides
  -   Fully connected layer with 120 nodes
  -   Fully connected layer with 84 nodes
  -   Fully connected softmax output layer with 10 nodes
-   All layers requiring initialization should initialize their kernels with the `he_normal` initialization method
-   All hidden layers requiring activation should use the `relu` activation function
-   you may `import tensorflow.keras as K`
-   Returns: a `K.Model` compiled to use `Adam` optimization (with default hyperparameters) and `accuracy` metrics
