from unittest import result
import numpy as np
from enum import Enum, auto


class ActivationFunctions(Enum):
    relu = auto()
    softmax = auto()


class Layer:
    def __init__(self, inputSize: int, neurons: int):
        self.neurons = neurons
        self.weights = 0.1 * np.random.randn(inputSize, neurons)
        self.biases = np.zeros((1, neurons))

    def forward(self, inputs: int, activationFunction=ActivationFunctions.relu):
        output = np.dot(inputs, self.weights) + self.biases
        self.output = self._pickActivationFunction(output, activationFunction)

    @staticmethod
    def _pickActivationFunction(inputs, activationFunction):
        if (activationFunction is ActivationFunctions.relu):
            return np.maximum(0, inputs)

        elif (activationFunction is ActivationFunctions.softmax):
            exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
            probabilities = exp_values / \
                np.sum(exp_values, axis=1, keepdims=True)
            return probabilities
