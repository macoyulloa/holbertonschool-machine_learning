#!/usr/bin/env python3
""" trains the model triple loss model """

from triplet_loss import TripletLoss
import tensorflow as tf


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
        # create the new model: training model
        A = np.random.uniform(0, 1, (2, 128))
        P = np.random.uniform(0, 1, (2, 128))
        N = np.random.uniform(0, 1, (2, 128))
        inputs = [A, P, N]
        output = TripletLoss(base_model.output)
        training_model = tf.keras.models.Model(inputs, output)
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
                x_t, y_train_oh,
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
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
            predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
            precision = true_positives / (predicted_positives + K.epsilon())
            recall = true_positives / (possible_positives + K.epsilon())
            f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
            return f1_val

        @staticmethod
        def accuracy(y_true, y_pred):
            """ define the accuracy """
            return roc_auc_score(yprobs, probs)
