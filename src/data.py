import numpy as np


def spiral(samples, classes):
    x = np.zeros((samples*classes, 2))
    y = np.zeros(samples*classes, dtype='uint8')

    for classNumber in range(classes):
        ix = range(samples*classNumber, samples*(classNumber+1))
        r = np.linspace(0.0, 1, samples)
        t = np.linspace(classNumber*4, (classNumber+1)*4,
                        samples) + np.random.randn(samples)*0.2
        x[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = classNumber

    return x, y


def vertical(samples, classes):
    X = np.zeros((samples*classes, 2))
    y = np.zeros(samples*classes, dtype='uint8')

    for class_number in range(classes):
        ix = range(samples*class_number, samples*(class_number+1))
        X[ix] = np.c_[np.random.randn(
            samples)*.1 + (class_number)/3, np.random.randn(samples)*.1 + 0.5]
        y[ix] = class_number

    return X, y
