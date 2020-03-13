Tasks
----

#### 0\. Sequential

Write a function `def build_model(nx, layers, activations, lambtha, keep_prob):` that builds a neural network with the Keras library:

-   `nx` is the number of input features to the network
-   `layers` is a list containing the number of nodes in each layer of the network
-   `activations` is a list containing the activation functions used for each layer of the network
-   `lambtha` is the L2 regularization parameter
-   `keep_prob` is the probability that a node will be kept for dropout
-   You are not allowed to use the `Input` class
-   Returns: the keras model

#### 1\. Input mandatory

Write a function `def build_model(nx, layers, activations, lambtha, keep_prob):` that builds a neural network with the Keras library:

-   `nx` is the number of input features to the network
-   `layers` is a list containing the number of nodes in each layer of the network
-   `activations` is a list containing the activation functions used for each layer of the network
-   `lambtha` is the L2 regularization parameter
-   `keep_prob` is the probability that a node will be kept for dropout
-   You are not allowed to use the `Sequential` class
-   Returns: the keras model

#### 2\. Optimize mandatory

Write a function `def optimize_model(network, alpha, beta1, beta2):` that sets up Adam optimization for a keras model with categorical crossentropy loss and accuracy metrics:

-   `network` is the model to optimize
-   `alpha` is the learning rate
-   `beta1` is the first Adam optimization parameter
-   `beta2` is the second Adam optimization parameter
-   Returns: `None`

#### 3\. One Hot mandatory

Write a function `def one_hot(labels, classes=None):` that converts a label vector into a one-hot matrix:

-   The last dimension of the one-hot matrix must be the number of classes
-   Returns: the one-hot matrix

#### 4\. Train mandatory

Write a function `def train_model(network, data, labels, batch_size, epochs, verbose=True, shuffle=False):` that trains a model using mini-batch gradient descent:

-   `network` is the model to train
-   `data` is a `numpy.ndarray` of shape `(m, nx)` containing the input data
-   `labels` is a one-hot `numpy.ndarray` of shape `(m, classes)` containing the labels of `data`
-   `batch_size` is the size of the batch used for mini-batch gradient descent
-   `epochs` is the number of passes through `data` for mini-batch gradient descent
-   `verbose` is a boolean that determines if output should be printed during training
-   `shuffle` is a boolean that determines whether to shuffle the batches every epoch. Normally, it is a good idea to shuffle, but for reproducibility, we have chosen to set the default to `False`.
-   Returns: the `History` object generated after training the model

#### 5\. Validate mandatory

Based on `4-train.py`, update the function `def train_model(network, data, labels, batch_size, epochs, validation_data=None, verbose=True, shuffle=False):` to also analyze validaiton data:

-   `validation_data` is the data to validate the model with, if not `None`

#### 6\. Early Stopping mandatory

Based on `5-train.py`, update the function `def train_model(network, data, labels, batch_size, epochs, validation_data=None, early_stopping=False, patience=0, verbose=True, shuffle=False):` to also train the model using early stopping:

-   `early_stopping` is a boolean that indicates whether early stopping should be used
    -   early stopping should only be performed if `validation_data` exists
        -   early stopping should be based on validation loss
	-   `patience` is the patience used for early stopping

#### 7\. Learning Rate Decay mandatory

Based on `6-train.py`, update the function `def train_model(network, data, labels, batch_size, epochs, validation_data=None, early_stopping=False, patience=0, learning_rate_decay=False, alpha=0.1, decay_rate=1, verbose=True, shuffle=False):` to also train the model with learning rate decay:

-   `learning_rate_decay` is a boolean that indicates whether learning rate decay should be used
    -   learning rate decay should only be performed if `validation_data` exists
        -   the decay should be performed using inverse time decay
	    -   the learning rate should decay in a stepwise fashion after each epoch
	        -   each time the learning rate updates, `Keras` should print a message
		-   `alpha` is the initial learning rate
		-   `decay_rate` is the decay rate


#### 8\. Save Only the Best mandatory

Based on `7-train.py`, update the function `def train_model(network, data, labels, batch_size, epochs, validation_data=None, early_stopping=False, patience=0, learning_rate_decay=False, alpha=0.1, decay_rate=1, save_best=False, filepath=None, verbose=True, shuffle=False):` to also save the best iteration of the model:

-   `save_best` is a boolean indicating whether to save the model after each epoch if it is the best
    -   a model is considered the best if its validation loss is the lowest that the model has obtained
    -   `filepath` is the file path where the model should be saved

#### 9\. Save and Load Model mandatory

Write the following functions:

-   `def save_model(network, filename):` saves an entire model:
    -   `network` is the model to save
        -   `filename` is the path of the file that the model should be saved to
	    -   Returns: `None`
	    -   `def load_model(filename):` loads an entire model:
	        -   `filename` is the path of the file that the model should be loaded from
		    -   Returns: the loaded model

#### 10\. Save and Load Weights mandatory

Write the following functions:

-   `def save_weights(network, filename, save_format='h5'):` saves a model's weights:
    -   `network` is the model whose weights should be saved
        -   `filename` is the path of the file that the weights should be saved to
	    -   `save_format` is the format in which the weights should be saved
	        -   Returns: `None`
		-   `def load_weights(network, filename):` loads a model's weights:
		    -   `network` is the model to which the weights should be loaded
		        -   `filename` is the path of the file that the weights should be loaded from
			    -   Returns: `None`


#### 11\. Save and Load Configuration mandatory

Write the following functions:

-   `def save_config(network, filename):` saves a model's configuration in JSON format:
    -   `network` is the model whose configuration should be saved
        -   `filename` is the path of the file that the configuration should be saved to
	    -   Returns: `None`
	    -   `def load_config(filename):` loads a model with a specific configuration:
	        -   `filename` is the path of the file containing the model's configuration in JSON format
		    -   Returns: the loaded model


#### 12\. Test mandatory

Write a function `def test_model(network, data, labels, verbose=True):` that tests a neural network:

-   `network` is the network model to test
-   `data` is the input data to test the model with
-   `labels` are the correct one-hot labels of `data`
-   `verbose` is a boolean that determines if output should be printed during the testing process
-   Returns: the loss and accuracy of the model with the testing data, respectively


#### 13\. Predict mandatory

Write a function `def predict(network, data, verbose=False):` that makes a prediction using a neural network:

-   `network` is the network model to make the prediction with
-   `data` is the input data to make the prediction with
-   `verbose` is a boolean that determines if output should be printed during the prediction process
-   Returns: the prediction for the data
