import numpy as np
from neuron import Neuron


class SimpleNeuralNetwork:
    '''
    A neural network with:
        - 2 inputs
        - a hidden layer with 2 neurons (h1, h2)
        - an output layer with 1 neuron (o1)
    Each neuron has the same weights and bias:
        w = [0, 1]
        b = 0
    '''
    def __init__(self):
        weights = np.array([0, 1])  # vector form of:  w1=0, w2=1
        bias = 0
        

        # My Neuron class, all with the same w & b
        self.h1 = Neuron(weights, bias)
        self.h2 = Neuron(weights, bias)
        self.o1 = Neuron(weights, bias)

    def feedforward(self, x):
        out_h1 = self.h1.feedforward(x)
        out_h2 = self.h2.feedforward(x)

        # The inputs for o1 are the outputs from the hidden layers h1, h2
        # Remember to use np.array, so to make it a vector and passable to feedforward
        # which only takes in 1 argument
        out_o1 = self.o1.feedforward(np.array([out_h1,out_h2]))

        return out_o1

network = SimpleNeuralNetwork()
x = np.array([2,3])
print(network.feedforward(x))