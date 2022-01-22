from unittest import result
import numpy as np
from enum import Enum, auto


class ActivationFunctions(Enum):
    relu = auto()
    softmax = auto()

# class ActivationFunctions:
#     def relu(self, inputs):
#         self.output = np.maximum(0, inputs)

#     def softmax(self, inputs):
#         exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
#         probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
#         self.output = probabilities

# TO-DO: implement functions


class Layer:
    def __init__(self, inputSize: int, neurons: int, activationFunction):
        self.weights = 0.1 * np.random.randn(inputSize, neurons)
        self.biases = np.zeros((1, neurons))
        self.test = self._pickActivationFunction(activationFunction)

    def forward(self, inputs: int):
        self.output = np.dot(inputs, self.weights) + self.biases

    @staticmethod
    def _pickActivationFunction(activationFunction):
        if (activationFunction is ActivationFunctions.relu):
            return 'relu'

        elif (activationFunction is ActivationFunctions.softmax):
            return 'softmax'
