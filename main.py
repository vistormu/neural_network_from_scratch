import numpy as np
from core.logger import Logger
from src import network as net
from src import data, optimizers


def main():

    # Logger
    logger = Logger(name='main')

    # Data
    samples, targets = data.vertical(samples=100, classes=3)
    nInputs = samples.shape[1]
    nClasses = 3

    # Layers and loss function
    # layer1 = net.Layer(inputSize=nInputs, neurons=3)
    # layer2 = net.Layer(inputSize=layer1.neurons, neurons=3)

    model = net.Model(nInputs=nInputs,
                      nOutputs=nClasses,
                      nLayers=2,
                      nNeuronsPerLayer=3,
                      lossFunction=net.CategoricalCrossEntropyLoss())

    optimizer = optimizers.VanillaSGD()

    iterations = 1001

    for epoch in range(iterations):

        model.forward(samples)
        model.calculateLoss(targets)
        model.backward(targets)

        for layer in model.layers:
            optimizer.optimize(layer)

    #     # Front-propagation
    #     layer1.forward(samples)
    #     layer2.forward(
    #         layer1.output, activationFunction=net.ActivationFunctions.softmax)

    #     lossFunction.calculate(layer2.output, targets)

    #     # Back-propagation
    #     lossFunction.backward(layer2.output, targets)
    #     layer2.backward(lossFunction.error, net.ActivationFunctions.softmax)
    #     layer1.backward(layer2.error)

        if not epoch % 100:
            logger.info('\nepoch: ', epoch, ', ',
                        'accuracy:', model.accuracy, ', ',
                        'loss: ', model.loss, ', ')

    #     optimizer.optimize(layer1)
    #     optimizer.optimize(layer2)


if __name__ == '__main__':
    main()
