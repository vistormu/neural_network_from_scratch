import numpy as np
import matplotlib.pyplot as plt


def createData(points, classes):
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')

    for classNumber in range(classes):
        ix = range(points*classNumber, points*(classNumber+1))
        r = np.linspace(0.0, 1, points)
        t = np.linspace(classNumber*4, (classNumber+1)*4,
                        points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = classNumber

    return X, y


class Layer:
    def __init__(self, inputs, neurons):
        self.weights = 0.1 * np.random.randn(inputs, neurons)
        self.biases = np.zeros((1, neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


X, y = createData(points=100, classes=3)

# plt.scatter(X[:, 0], X[:, 1])
# plt.show()

# plt.scatter(X[:, 0], X[:, 1], c=y, cmap="brg")
# plt.show()

layer1 = Layer(2, 3)
activation1 = Activation_ReLU()

layer2 = Layer(3, 3)
activation2 = Activation_Softmax()

layer1.forward(X)
activation1.forward(layer1.output)

layer2.forward(activation1.output)
activation2.forward(layer2.output)

print(activation2.output)

# TO-DO: estan mal las dimensiones
# P.6 31:32
