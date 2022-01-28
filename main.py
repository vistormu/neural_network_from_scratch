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
    layer1 = net.Layer(nInputs=nInputs, nNeurons=3)
    layer2 = net.Layer(nInputs=layer1.nNeurons, nNeurons=3,
                       activationFunction=net.ActivationFunctions.softmax)
    lossFunction = net.CategoricalCrossEntropyLoss()

    model = net.Model(nInputs=nInputs,
                      nOutputs=nClasses,
                      nLayers=2,
                      nNeuronsPerLayer=3,
                      lossFunction=net.CategoricalCrossEntropyLoss())

    model.dump()

    optimizer = optimizers.VanillaSGD()

    iterations = 0

    for epoch in range(iterations):

        model.forward(samples)
        model.calculateLoss(targets)
        model.backward(targets)

        for layer in model.layers:
            optimizer.optimize(layer)

        # Front-propagation
        # layer1.forward(samples)
        # layer2.forward(layer1.output)

        # lossFunction.calculate(layer2.output, targets)

        # # Back-propagation
        # lossFunction.backward(layer2.output, targets)
        # layer2.backward(lossFunction.error)
        # layer1.backward(layer2.error)

        if not epoch % 100:
            logger.info('\nepoch: ', epoch, ', ',
                        'accuracy:', model.accuracy, ', ',
                        'loss: ', model.loss, ', ')

    #     optimizer.optimize(layer1)
    #     optimizer.optimize(layer2)


if __name__ == '__main__':
    main()
