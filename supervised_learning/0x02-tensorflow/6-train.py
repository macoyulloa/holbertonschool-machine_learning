#!/usr/bin/env python3
"""traning the model that buils, trains, and saves"""

import tensorflow as tf


calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes,
          activations, alpha, iterations, save_path="/tmp/model.ckpt"):
    """saving the neural network"""
    graph = tf.graph()
    with graph.as_default():
        x, y = create_placeholders(X_train.shape[0], Y_train.shape[0])
        y_pred = forward_prop(x, layer_sizes, activations)
        accuracy = calculate_accuracy(y, y_pred)
        loss = calculate_loss(y, y_pred)
        train_op = create_train_op(loss, alpha)
        tf.add_to_collecton("my_colection", x, y, y_pred, accuracy, loss)
    saver = tf.train.Saver()
    with tf.Session(graph=graph) as session:
        for i in range(iterations):
            session.run(accuracy)
            session.run(loss)
            session.run(train_op)

            counter += 1
            if (i == 0) or (counter == 100) or (i == iterations-1):
                print("After {} iterations").format(i)
                print("Training Cost: {}".format(loss))
                counter = 0
        save_path = saver.save(session, save_path)
