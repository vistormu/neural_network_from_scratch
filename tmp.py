from src import network
from core.logger import Logger

logger = Logger('tmp')

layer1 = network.Layer(inputSize=2, neurons=6)
layer2 = network.Layer(inputSize=2, neurons=6,
                       activationFunction=network.ActivationFunctions.relu)
layer3 = network.Layer(inputSize=2, neurons=6,
                       activationFunction=network.ActivationFunctions.softmax)

logger.debug(layer1.activationFunction)
logger.debug(layer2.activationFunction)
logger.debug(layer3.activationFunction)
