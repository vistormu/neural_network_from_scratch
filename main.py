from core.logger import Logger
from src import network as net
from src.data import Data
from src.loss_functions import CategoricalCrossEntropyLoss


def main():

    logger = Logger(name='main')

    x, y = Data.vertical(samples=100, classes=3)

    inputSize = x.shape[1]
    input = x

    layer1 = net.Layer(inputSize=inputSize, neurons=3)
    layer2 = net.Layer(inputSize=layer1.neurons, neurons=3)

    layer1.forward(input)
    layer2.forward(
        layer1.output, activationFunction=net.ActivationFunctions.softmax)

    loss = CategoricalCrossEntropyLoss.calculate(layer2.output, y)

    logger.debug('output: ', layer2.output[:5])
    logger.debug('loss: ', loss)


if __name__ == '__main__':
    main()
