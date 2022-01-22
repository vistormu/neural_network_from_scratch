from src import network
from core.logger import Logger

logger = Logger('tmp')

layer = network.Layer(inputSize=2, neurons=6,
                      activationFunction=network.ActivationFunctions.softmax)

logger.debug(layer.test)
