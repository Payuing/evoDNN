"""
A module to implement the SGD for a feedforward neural network.
Gradients are calculated using backpropagation.
"""
from __future__ import absolute_import
from __future__ import print_function
import random
import numpy as np
import math
import csv
import sys

from sklearn.metrics import mean_squared_error
from math import sqrt

#### Miscellaneous functions
"""The sigmoid function."""
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

"""Derivative of the sigmoid function"""
def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

""""Root Mean Squared Error"""
def RMSE(y_predicted, y_true):
    return sqrt(mean_squared_error(y_true, y_predicted))

""""Multiclass Logloss"""
def multiclass_LOGLOSS(y_predicted, y_true):
    logloss_value = 0.0
    for i in range(y_predicted.shape[0]):
        for j in range(y_predicted.shape[1]):
            considered_value = min(max(y_predicted[i][j], 1.0E-15), 1.0-1.0E-15) # IEEE 754
            logloss_value += y_true[i][j]*math.log(considered_value)
    logloss_value *= -(1.0/y_predicted.shape[0])
    return logloss_value

""""Softmax"""
# x is a numpy array
def softmax(x):
    softmax_layer = np.zeros((x.shape[0], x.shape[1]))
    for i in range(x.shape[0]):
        shiftx = x[i] - np.max(x[i])
        exponent = np.exp(shiftx)
        sum_values = np.sum(exponent)
        softmax_layer[i] = exponent/sum_values
    return softmax_layer

"""Multi-class logarithmic loss funciton per class"""
def multiclass_LOGLOSS(y_predicted, y_true):
	logloss_value = 0.0
	for i in range(y_predicted.shape[0]):
		for j in range(y_predicted.shape[1]):
			considered_value = min(max(y_predicted[i][j], EPS), 1.0-EPS)
			logloss_value +=  y_true[i][j]* math.log(considered_value)

	logloss_value *= -(1.0/y_predicted.shape[0])
	return logloss_value

class Network(object):

    def __init__(self, sizes, type='classification'):
        """
        The list "sizes" contains the number of nodes in the respecitve layers.
        [3, 4, 5] is a three-layer network, with 1st layer containing 3 nodes etc.
        The biases and weights are initialized using Gaussian distribution w/ mean 0,
        and variance 1. First layer is the input layer and no bias.
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                            for x, y in zip(sizes[:-1], sizes[1:])] # the ordering make it easy to cal wa+b
        self.type = type
        if (self.type != 'regression') and (self.type != 'classification'):
            raise ValueError("Error: type must be \"regression\" or \"classification\"")

        if (self.type == 'regression'):
            self.output_function = sigmoid
        elif (self.type == 'classification'):
            self.output_function = softmax

    def feedforward(self, a):
        """Return the output of the network"""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, early_stopping, mini_batch_size, eta, test_data=None, test_data_len=None, test_true=None, verbose=1):
        """
        Training the network using stochastic gradient descent.
        "training_data" is a list of tuples of numpy array; x is training inputs and y is the desired outputs.
        If "test_data" is provided then the network will be evaluated against the test data
        after each epoch, and partial progress print out
        "eta" is the learning rate
        """
        if test_data: n_test = len(test_data)
        n = len(training_data)
        evaluate_timer = 0
        early_stopping_flag = 0
        logloss = float('inf')
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if ((j+1)%10 == 0) & (verbose == 1):
                if test_data:
                    print("Epoch {}: {} / {}".format(j+1, self.evaluate(test_data), n_test))
                    if( evaluate_timer <= int((early_stopping / 10))):
                        pred_probabilities = self.predict_proba(test_data)
                        temp = multiclass_LOGLOSS(pred_probabilities, test_true)
                        if (logloss > temp):
                            logloss = temp
                            evaluate_timer = 0
                        else:
                            evaluate_timer += 1
                    else:
                        early_stopping_flag = 1
                else:
                    print("Epoch {} completes".format_map(j))
            if (early_stopping_flag == 1):
                break

    def update_mini_batch(self, mini_batch, eta):
        """
        Update the network's weights and biases by applying gradient descent using backpropagation
        to a single batch.
        The "mini_batch" is a list of tuples "(x,y)."
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """ Return a tuple "(nabla_b, nabla_w)" representing the gradient
        for the cost function C_x. "nabla_b" and "nabla_w" are layer-by-layer
        lists of numpy arrays.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta  = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # l = 1 is the last layers, l = 2 is the second-last layer
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def cost_derivative(self, output_activations, y):
        """"Return the vector of partial derivatives of Mean Squared Error"""
        return (output_activations-y)

    def evaluate(self, test_data):
        """
        test_data are list contaning 2-tuples (x,y)
        x is a (x1,1) numpy array
        y is a (y1,1) numpy array
        """
        test_results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                            for (x,y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def predict_proba(self, test_data):
        """
        test_data are list contaning 2-tuples (x,y)
        x is a (x1,1) numpy array
        y is a (y1,1) numpy array
        """
        if (self.type == 'regression'):
            raise ValueError("Regression cannot predict probabilities")
            exit()
        test_outputs = []
        for (x,y) in test_data:
            x_output = self.feedforward(x)
            test_output = np.reshape(x_output,(y.shape[0]))
            test_output = test_output.tolist() # convert to list
            test_outputs.append(test_output)

        test_outputs_array = np.array(test_outputs) # list to np array is much fast
        return self.output_function(test_outputs_array)
