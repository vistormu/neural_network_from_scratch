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

    # Model
    model = net.Model(nInputs=nInputs,
                      nOutputs=nClasses,
                      nLayers=2,
                      nNeuronsPerLayer=3,
                      lossFunction=net.CategoricalCrossEntropyLoss(),
                      optimizer=optimizers.VanillaSGD())

    iterations = 1001

    for epoch in range(iterations):

        model.forward(samples)
        model.calculateLoss(targets)
        model.backward(targets)
        model.optimize()

        if not epoch % 200:
            logger.info('\nepoch: ', epoch, ', ',
                        'accuracy:', model.accuracy, ', ',
                        'loss: ', model.loss, ', ')


if __name__ == '__main__':
    main()
