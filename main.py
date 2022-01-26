from re import A
from core.logger import Logger
from src import network as net
from src import data


def main():

    logger = Logger(name='main')

    samples, targets = data.vertical(samples=100, classes=3)

    inputSize = samples.shape[1]

    layer1 = net.Layer(inputSize=inputSize, neurons=3)
    layer2 = net.Layer(inputSize=layer1.neurons, neurons=3)

    layer1.forward(samples)
    layer2.forward(
        layer1.output, activationFunction=net.ActivationFunctions.softmax)

    loss = net.CategoricalCrossEntropyLoss.calculate(layer2.output, targets)

    logger.debug('output:\n', layer2.output[:5])
    logger.debug('loss:\n', loss)

    layer2.backward(layer2.output, targets,
                    activationFunction=net.ActivationFunctions.softmax)
    layer1.backward(layer2.error)

    logger.debug('layer1 dweights:\n', layer1.dweights)
    logger.debug('layer1 dbiases:\n', layer1.dbiases)
    logger.debug('layer2 dweights:\n', layer2.dweights)
    logger.debug('layer2 dbiases:\n', layer2.dbiases)


if __name__ == '__main__':
    main()
