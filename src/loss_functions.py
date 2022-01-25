import numpy as np


class CategoricalCrossEntropyLoss:
    @staticmethod
    def calculate(predictions, targets):
        predictions = np.clip(predictions, 1e-7, 1-1e-7)

        if len(targets.shape) == 1:
            correctConfidences = predictions[range(len(predictions)), targets]
        elif len(targets.shape) == 2:
            correctConfidences = np.sum(predictions*targets, axis=1)

        likelihoods = -np.log(correctConfidences)
        loss = np.mean(likelihoods)

        return loss
