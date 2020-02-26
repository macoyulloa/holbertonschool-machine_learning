#!/usr/bin/env python3
"""evaluated the saved neural network"""

import tensorflow as tf


calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
train = __import__('6-train').train


def evaluate(X, Y, save_path):
    """evaluated a saved neural network"""
    saver = tf.train.Saver()
    with tf.Session() as session:
        saver.restore(session, save_path)
        session.run([accuracy, loss])
        print("Test Accuracy: ", accuracy)
        print("Test Cost: ", loss)
