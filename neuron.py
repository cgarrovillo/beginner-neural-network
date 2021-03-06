import numpy as np

def sigmoid(x):
    # Simple Logistic Sigmoid Function
    # https://en.wikipedia.org/wiki/Logistic_function
    # f(x) = 1 / (1+ e^(-x))
    return 1 / (1 + np.exp(-x))

class Neuron:
    # class-private instance and class variables
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedforward(self, inputs):
        # Weight inputs, add bias, then use the activation function
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)
