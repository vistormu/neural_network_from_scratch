from os import stat
import numpy as np
import enum


class ActivationFunctions(enum.Enum):
    relu = enum.auto()
    softmax = enum.auto()


class _ActivationRelu:
    @staticmethod
    def forward(inputs):
        return np.maximum(0, inputs)

    @staticmethod
    def backward(inputs):
        output = inputs.copy()
        output[inputs <= 0] = 0

        return output


class _ActivationSoftmax:
    @staticmethod
    def forward(inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return probabilities

    @staticmethod
    def backward(predictions, targets):

        if len(targets.shape) == 2:
            targets = np.argmax(targets, axis=1)

        output = predictions.copy()

        output[range(len(predictions)), targets] -= 1
        output = output / len(predictions)

        return output


class CategoricalCrossEntropyLoss:
    @staticmethod
    def calculate(predictions, targets):
        predictions = np.clip(predictions, 1e-7, 1-1e-7)

        if len(targets.shape) == 1:
            correctConfidences = predictions[range(len(predictions)), targets]
        elif len(targets.shape) == 2:
            correctConfidences = np.sum(predictions*targets, axis=1)

        likelihoods = -np.log(correctConfidences)
        loss = np.mean(likelihoods)

        return loss


class Layer:
    def __init__(self, inputSize: int, neurons: int):
        self.neurons = neurons
        self.weights = 0.1 * np.random.randn(inputSize, neurons)
        self.biases = np.zeros((1, neurons))

    def forward(self, inputs, activationFunction=ActivationFunctions.relu):
        self.inputs = inputs
        output = np.dot(inputs, self.weights) + self.biases
        self.output = self._pickForwardFunction(activationFunction, output)

    def backward(self, error, targets=None, activationFunction=ActivationFunctions.relu):
        activationBackward = self._pickBackwardFunction(
            activationFunction, error, targets)
        self.dbiases = np.sum(activationBackward, axis=0, keepdims=True)
        self.dweights = np.dot(self.inputs.T, error)
        self.error = np.dot(error, self.weights.T)

    @staticmethod
    def _pickForwardFunction(activationFunction, inputs):
        if activationFunction is ActivationFunctions.relu:
            return _ActivationRelu.forward(inputs)

        elif activationFunction is ActivationFunctions.softmax:
            return _ActivationSoftmax.forward(inputs)

    @staticmethod
    def _pickBackwardFunction(activationFunction, error, targets):
        if activationFunction is ActivationFunctions.relu:
            return _ActivationRelu.backward(error)

        elif activationFunction is ActivationFunctions.softmax:
            return _ActivationSoftmax.backward(error, targets)
