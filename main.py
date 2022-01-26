import numpy as np
from core.logger import Logger
from src import network as net
from src import data


def main():

    # Logger
    logger = Logger(name='main')

    # Data
    samples, targets = data.vertical(samples=100, classes=3)
    inputSize = samples.shape[1]

    # Layers and loss function
    layer1 = net.Layer(inputSize=inputSize, neurons=3)
    layer2 = net.Layer(inputSize=layer1.neurons, neurons=3)
    lossFunction = net.CategoricalCrossEntropyLoss()

    # Front-propagation
    layer1.forward(samples)
    layer2.forward(
        layer1.output, activationFunction=net.ActivationFunctions.softmax)

    lossFunction.calculate(layer2.output, targets)

    # Back-propagation
    lossFunction.backward(layer2.output, targets)
    layer2.backward(lossFunction.error, net.ActivationFunctions.softmax)
    layer1.backward(layer2.error)


if __name__ == '__main__':
    main()
