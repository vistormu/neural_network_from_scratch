import numpy as np

from core.logger import Logger
from src import data, network


def main():

    logger = Logger(name='main')

    x, y = data.create(points=100, classes=3)

    inputSize = x.shape[1]
    input = x[0:3, :]

    layer = network.Layer(inputSize=inputSize, neurons=6)
    layer.forward(input)


if __name__ == '__main__':
    main()
