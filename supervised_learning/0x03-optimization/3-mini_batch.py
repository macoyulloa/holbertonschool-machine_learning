#!/usr/bin/env python3
"""Optimization tasks"""

import numpy as np
import tensorflow as tf

shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid,
                     batch_size=32, epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """trains a neural network nodel using mini-bash gradient desc"""
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(load_path + ".meta")
        saver.restore(sess, load_path)

        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]
        accuracy = tf.get_collection("accuracy")[0]
        loss = tf.get_collection("loss")[0]
        train_op = tf.get_collection("train_op")[0]

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        sess.run(init)
        for i in range(epochs + 1):
            cost_train = sess.run(loss,
                                  feed_dict={x: X_train, y: Y_train})
            accuracy_train = sess.run(accuracy,
                                      feed_dict={x: X_train, y: Y_train})
            cost_valid = sess.run(loss,
                                  feed_dict={x: X_valid, y: Y_valid})
            accuracy_valid = sess.run(accuracy,
                                      feed_dict={x: X_valid, y: Y_valid})

            print("After {} epochs:".format(i))
            print("\tTraining Cost: {}".format(cost_train))
            print("\tTraining Accuracy: {}".format(accuracy_train))
            print("\tValidation Cost: {}".format(cost_valid))
            print("\tValidation Accuracy: {}".format(accuracy_valid))

            if i < epochs:
                X_shuffle, Y_shuffle = shuffle_data(X_train, Y_train)
                train_op = sess.run(train_op,
                                    feed_dict={x: X_train, y: Y_train})
        save_path = saver.save(sess, save_path)
    return save_path
