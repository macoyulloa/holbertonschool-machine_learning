#!/usr/bin/env python3
""" trains the model triple loss model """

from triplet_loss import TripletLoss
import tensorflow as tf
import temsorflow.keras as K


class TrainModel():
    """trains the model for face verification using triplet loss
    """
    def __init__(self, model_path, alpha):
        """ initialization of the variables
        Arg:
            - model_path: parth to the base face verification embedding mode
                 - loads model: tf.keras.utils.CustomObjectScope({'tf': tf}):
                 - saves this model as the public instance method base_model
            - alpha: is the alpha value used to calculate the triplet loss
        Creates a new model:
            inputs: [A, P, N]
                 numpy.ndarrys with the anchor, positive and negatives images
            outputs: the triplet losses of base_model
            compiles the model with Adam optimization and no losses
            save this model as the public instance method training_model
        """
        with tf.keras.utils.CustomObjectScope({'tf': tf}):
            self.base_model = tf.keras.models.load_model(model_path)
        self.alpha = alpha
        # define the tensors for the three input images
        A = K.Input(input_shape, name="A")
        P = K.Input(input_shape, name="P")
        N = K.Input(input_shape, name="N")
        inputs = [A, P, N]
        # generate the encoding (feature vectors) for the three images
        encoded_a = self.base_model(A)
        encoded_p = self.base_model(P)
        encoded_n = self.base_model(N)
        encoded = [encoded_a, encoded_p, encoded_n]
        # TripletLoss layer
        loss_layer = TripletLossLayer(alpha=alpha,
                                      name='triplet_loss_layer')(encoded)
        # create the new model: training model, connect inputs, outputs
        training_model = K.models.Model(inputs, loss_layer)
        training_model.compile(optimizer='Adam')
        training_model.save()

    def train(self, triplets, epochs=5, batch_size=32,
              validation_split=0.3, verbose=True):
        """training method
        Arg:
        - triplets: list containing the inputs to self.training_model
        - epochs: number of epochs to train for
        - batch_size: is the batch size for training
        - validation_split: is the validation split for training
        - verbose: is a boolean that sets the verbosity mode
        Returns: the History output from the training
        """
        history = self.training_model.fit(
            triplets, y_train_oh,
            validation_split=validation_split
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose)
        return history

    def save(self, save_path):
        """saves the base embedding model
        Arg:
        save_path: path save the model
        Returns: the saved model
        """
        return self.training_model.save(save_path)

    @staticmethod
    def f1_score(y_true, y_pred):
        """ define the f1 score """
        true_positives = K.backend.sum(K.backend.round(
            K.backend.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.backend.sum(K.backend.round(
            K.backend.clip(y_true, 0, 1)))
        predicted_positives = K.backend.sum(K.backend.round(
            K.backend.clip(y_pred, 0, 1)))
        precision = true_positives / (
            predicted_positives + K.backend.epsilon())
        recall = true_positives / (
            possible_positives + K.backend.epsilon())
        f1_val = 2*(precision * recall)/(
            precision + recall + K.backend.epsilon())

        return f1_val

    @staticmethod
    def accuracy(y_true, y_pred):
        """ define the accuracy """
        return roc_auc_score(y_true, y_pred)

    def best_tau(self, images, identities, thresholds):
        """
        """
        y_true = []
        y_pred = []
        tau = []
        acc = self.accuracy(y_true, y_pred)
        f1_score = self.f1_score(y_true, y_pred)

        return (tau, f1_score, acc)
