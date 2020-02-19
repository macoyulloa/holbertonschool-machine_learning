#!/usr/bin/env python3
" defines a single neuron performing binary classification "

import numpy as np
import matplotlib.pyplot as plt


class Neuron():
    " single neuron binary classification "

    def __init__(self, nx):
        " initialized variables "
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        " gett the weight as W "
        return self.__W

    @property
    def b(self):
        " gett the bias as b"
        return self.__b

    @property
    def A(self):
        " get the activation function as A"
        return self.__A

    def forward_prop(self, X):
        " return the forward propagation of the neuron, using sigmoid "
        matrix_mul = (np.matmul(self.__W, X)) + self.__b
        self.__A = 1 / (1 + np.exp(- matrix_mul))
        return self.__A

    def cost(self, Y, A):
        " cost of the classification model, logistic regression "
        m = Y.shape[1]
        cost = (-1/m) * np.sum(Y * np.log(A) + (1-Y) * (np.log(1.0000001 - A)))
        return cost

    def evaluate(self, X, Y):
        " evaluates the neuron's predictions "
        self.__A = self.forward_prop(X)
        evalu = np.where(self.__A >= 0.5, 1, 0)
        cost_y = self.cost(Y, self.__A)
        return (evalu, cost_y)

    def gradient_descent(self, X, Y, A, alpha=0.05):
        " calculate the gradient descent on the neuron, derive "
        m = Y.shape[1]
        dz = A - Y
        dw = (1/m) * np.matmul(X, dz.T)
        db = (1/m) * np.sum(dz)
        self.__W = self.__W - (alpha * dw).T
        self.__b = self.__b - alpha * db

    def train(self, X, Y, iterations=5000,
              alpha=0.05, verbose=True, graph=True, step=100):
        " Training the neuron with the graph "
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations < 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")
        if verbose is True or graph is True:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step < 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        cost_list = []
        for i in range(iterations + 1):
            self.__A = self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A, alpha)
            cost = self.cost(Y, self.__A)
            if verbose is True:
                if (i % step == 0 or step == iterations):
                    print("Cost after {} iterations: {}".format(i, cost))
                    if i < iterations:
                        cost_list.append(cost)
        if graph is True:
            x_list = np.arange(0, iterations, step)
            y_list = cost_list
            plt.plot(x_list, y_list)
            plt.title('Training Cost')
            plt.xlabel('iterations')
            plt.ylabel('cost')
            plt.show()
        return self.evaluate(X, Y)
