import numpy as np

from core.logger import Logger


logger = Logger(name='main')


def createData(points, classes):
    x = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')

    for classNumber in range(classes):
        ix = range(points*classNumber, points*(classNumber+1))
        r = np.linspace(0.0, 1, points)
        t = np.linspace(classNumber*4, (classNumber+1)*4,
                        points) + np.random.randn(points)*0.2
        x[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = classNumber

    return x, y


class Layer:
    def __init__(self, inputSize, neurons):
        self.weights = 0.1 * np.random.randn(inputSize, neurons)
        self.biases = np.zeros((1, neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


def main():

    x, y = createData(points=100, classes=3)

    inputSize = x.shape[1]

    layer = Layer(inputSize=inputSize, neurons=6)
    layer.forward(x[0, :])

    logger.debug(layer.weights)
    logger.debug(layer.biases)
    logger.debug(layer.output)


if __name__ == '__main__':
    main()

# TO-DO: change logger
