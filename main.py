import numpy as np

from core.logger import Logger
from src import data, network


class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss


class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods


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

    loss_function = Loss_CategoricalCrossEntropy()
    loss = loss_function.calculate(layer2.output, y)

    print('Loss: ', loss)


if __name__ == '__main__':
    main()
