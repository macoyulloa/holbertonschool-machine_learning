0x05. Generative Adversarial Networks
=====================================

Tasks
-----

#### 0\. Generator

Function `def generator(Z):` that creates a simple generator network for MNIST digits:

-   `Z` is a `tf.tensor` containing the input to the generator network
-   The network should have two layers:
    -   the first layer should have 128 nodes and use relu activation with name `layer_1`
    -   the second layer should have 784 nodes and use a sigmoid activation with name `layer_2`
-   All variables in the network should have the scope `generator` with `reuse=tf.AUTO_REUSE`
-   Returns `X`, a `tf.tensor` containing the generated image


#### 1\. Discriminator 

Function `def discriminator(X):` that creates a discriminator network for MNIST digits:

-   `X` is a `tf.tensor` containing the input to the discriminator network
-   The network should have two layers:
    -   the first layer should have 128 nodes and use relu activation with name `layer_1`
    -   the second layer should have 1 node and use a sigmoid activation with name `layer_2`
-   All variables in the network should have the scope `discriminator` with `reuse=tf.AUTO_REUSE`
-   Returns `Y`, a `tf.tensor` containing the classification made by the discriminator

#### 2\. Train Discriminator

Function `def train_discriminator(Z, X):` that creates the loss tensor and training op for the discriminator:

-   `Z` is the `tf.placeholder` that is the input for the generator
-   `X` is the `tf.placeholder` that is the real input for the discriminator
-   You can use the following imports:
    -   `generator = __import__('0-generator').generator`
    -   `discriminator = __import__('1-discriminator').discriminator`
-   The discriminator should minimize the negative minimax loss
-   The discriminator should be trained using Adam optimization
-   The generator should NOT be trained
-   Returns: `loss, train_op`
    -   `loss` is the discriminator loss
    -   `train_op` is the training operation for the discriminator


#### 3\. Train Generator

Function `def train_generator(Z):` that creates the loss tensor and training op for the generator:

-   `Z` is the `tf.placeholder` that is the input for the generator
-   `X` is the `tf.placeholder` that is the input for the discriminator
-   You can use the following imports:
    -   `generator = __import__('0-generator').generator`
    -   `discriminator = __import__('1-discriminator').discriminator`
-   The generator should minimize the negative modified minimax loss
-   The generator should be trained using Adam optimization
-   The discriminator should NOT be trained
-   Returns: `loss, train_op`
    -   `loss` is the generator loss
    -   `train_op` is the training operation for the generator


#### 4\. Sample Z

Function `def sample_Z(m, n):` that creates input for the generator:

-   `m` is the number of samples that should be generated
-   `n` is the number of dimensions of each sample
-   All samples should be taken from a random uniform distribution within the range `[-1, 1]`
-   Returns: `Z`, a `numpy.ndarray` of shape `(m, n)` containing the uniform samples


#### 5\. Train GAN

Function `def train_gan(X, epochs, batch_size, Z_dim, save_path='/tmp'):` that trains a GAN:

-   `X` is a `np.ndarray` of shape `(m, 784)` containing the real data input
    -   `m` is the number of real data samples
-   `epochs` is the number of epochs that the each network should be trained for
-   `batch_size` is the batch size that should be used during training
-   `Z_dim` is the number of dimensions for the randomly generated input
-   `save_path` is the path to save the trained generator
    -   Create the `tf.placeholder` for `Z` and add it to the graph's collection
-   The discriminator and generator training should be altered after one epoch
-   You can use the following imports:
    -   `train_generator = __import__('2-train_generator').train_generator`
    -   `train_discriminator = __import__('3-train_discriminator').train_discriminator`
    -   `sample_Z = __import__('4-sample_Z').sample_Z`
