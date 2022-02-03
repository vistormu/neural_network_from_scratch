from os import stat
from turtle import forward
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
    def backward(error, inputs):
        activationError = error.copy()
        activationError[inputs <= 0] = 0

        return activationError


class _ActivationSoftmax:
    @staticmethod
    def forward(inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return probabilities


class CategoricalCrossEntropyLoss:
    def calculate(self, predictions, targets):
        predictions = np.clip(predictions, 1e-7, 1-1e-7)

        if len(targets.shape) == 2:
            targets = np.argmax(targets, axis=1)

        correctConfidences = predictions[range(len(predictions)), targets]

        likelihoods = -np.log(correctConfidences)
        self.loss = np.mean(likelihoods)
        self.accuracy = np.mean(np.argmax(predictions, axis=1) == targets)

    def backward(self, predictions, targets):
        if len(targets.shape) == 2:
            targets = np.argmax(targets, axis=1)

        output = predictions.copy()

        output[range(len(predictions)), targets] -= 1
        self.error = output / len(predictions)


class Layer:
    def __init__(self, nInputs, nNeurons, activationFunction=ActivationFunctions.relu):
        self.nNeurons = nNeurons
        self.weights = 0.1 * np.random.randn(nInputs, nNeurons)
        self.biases = np.zeros((1, nNeurons))
        self.activationFunction = activationFunction

    def forward(self, inputs):
        self.inputs = inputs
        output = np.dot(inputs, self.weights) + self.biases
        self.output = self._pickForwardFunction(
            self.activationFunction, output)

    def backward(self, error):
        activationError = self._pickBackwardFunction(
            self.activationFunction, error, self.output)

        self.dWeights = np.dot(self.inputs.T, activationError)
        self.dBiases = np.sum(activationError, axis=0, keepdims=True)
        self.error = np.dot(activationError, self.weights.T)

    @ staticmethod
    def _pickForwardFunction(activationFunction, inputs):
        if activationFunction is ActivationFunctions.relu:
            return _ActivationRelu.forward(inputs)

        elif activationFunction is ActivationFunctions.softmax:
            return _ActivationSoftmax.forward(inputs)

    @ staticmethod
    def _pickBackwardFunction(activationFunction, error, inputs):
        if activationFunction is ActivationFunctions.relu:
            return _ActivationRelu.backward(error, inputs)

        if activationFunction is ActivationFunctions.softmax:
            return error


class Model:
    def __init__(self, nInputs, nOutputs, nLayers, nNeuronsPerLayer, lossFunction, optimizer):
        self.nInputs = nInputs
        self.nOutputs = nOutputs
        self.nLayers = nLayers
        self.nNeuronsPerLayer = nNeuronsPerLayer
        self.lossFunction = lossFunction
        self.optimizer = optimizer

        self._initLayers()

    def _initLayers(self):
        self.layers = []
        firstLayer = Layer(self.nInputs, self.nNeuronsPerLayer)
        self.layers.append(firstLayer)
        for _ in range(1, self.nLayers):
            layer = Layer(self.nNeuronsPerLayer, self.nNeuronsPerLayer)
            self.layers.append(layer)
        lastLayer = Layer(self.nNeuronsPerLayer, self.nOutputs,
                          activationFunction=ActivationFunctions.softmax)
        self.layers.append(lastLayer)

    def forward(self, samples):
        input = samples
        for i in range(len(self.layers)):
            self.layers[i].forward(input)
            input = self.layers[i].output

    def calculateLoss(self, targets):
        predictions = self.layers[-1].output
        self.lossFunction.calculate(predictions, targets)
        self.loss = self.lossFunction.loss
        self.accuracy = self.lossFunction.accuracy

    def backward(self, targets):
        predictions = self.layers[-1].output
        self.lossFunction.backward(predictions, targets)
        error = self.lossFunction.error
        for i in range(len(self.layers)-1, -1, -1):
            self.layers[i].backward(error)
            error = self.layers[i].error

    def optimize(self):
        for layer in self.layers:
            self.optimizer.optimize(layer)

    def dump(self):
        print('x'*self.nInputs)
        print('')
        for i in range(self.nLayers):
            print('x'*self.nNeuronsPerLayer)
            print('')
        print('x'*self.nOutputs)
