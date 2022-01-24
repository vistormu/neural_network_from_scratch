import numpy as np

from core.logger import Logger
from src import data, network
from src.loss_functions import CategoricalCrossEntropyLoss


def main():

    logger = Logger(name='main')

    x, y = data.create(points=100, classes=3)

    inputSize = x.shape[1]
    input = x

    layer1 = network.Layer(inputSize=inputSize, neurons=3)
    layer2 = network.Layer(inputSize=layer1.neurons, neurons=3)

    layer1.forward(input)
    layer2.forward(
        layer1.output, activationFunction=network.ActivationFunctions.softmax)

    loss = CategoricalCrossEntropyLoss().calculate(layer2.output, y)

    logger.debug('output: ', layer2.output[:5])
    logger.debug('loss: ', loss)


if __name__ == '__main__':
    main()
