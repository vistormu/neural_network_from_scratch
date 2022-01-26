import numpy as np
from core.logger import Logger
from src import network as net
from src import data


def main():

    logger = Logger(name='main')

    samples, targets = data.vertical(samples=100, classes=3)

    samples = np.array([[0., 0.],
                        [-0.09183642, -0.23252112],
                        [-0.43380709,  0.248619],
                        [0.60314021,  0.44578233],
                        [-0.90731661, -0.42044805],
                        [-0.,         -0.],
                        [-0.048897,    0.24517154],
                        [0.4992601,  -0.02719096],
                        [-0.73157048,  0.16524111],
                        [0.98796114, -0.15470223],
                        [0.,          0.],
                        [-0.0814754,  -0.23635092],
                        [0.03100753,  0.49903761],
                        [0.71064911, -0.23974536],
                        [-0.89924661,  0.43744205]])

    targets = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2])

    inputSize = samples.shape[1]

    layer1 = net.Layer(inputSize=inputSize, neurons=3)
    layer2 = net.Layer(inputSize=layer1.neurons, neurons=3)
    lossFunction = net.CategoricalCrossEntropyLoss()

    layer1.weights = np.zeros([inputSize, 3]) + 0.1
    layer2.weights = np.zeros([layer1.neurons, 3]) + 0.1

    layer1.forward(samples)
    layer2.forward(
        layer1.output, activationFunction=net.ActivationFunctions.softmax)

    lossFunction.calculate(layer2.output, targets)

    logger.debug('output:\n', layer2.output[:5])
    logger.debug('loss:\n', lossFunction.loss)

    lossFunction.backward(layer2.output, targets)
    layer2.backward(lossFunction.error, net.ActivationFunctions.softmax)
    layer1.backward(layer2.error)

    logger.debug('layer1 dweights:\n', layer1.dweights)
    logger.debug('layer1 dbiases:\n', layer1.dbiases)
    logger.debug('layer2 dweights:\n', layer2.dweights)
    logger.debug('layer2 dbiases:\n', layer2.dbiases)


if __name__ == '__main__':
    main()
