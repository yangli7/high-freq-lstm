from keras.models import Sequential
from .model import LSTMs
import numpy as np


#class epochsTrain(LSTMs):


def create_dataSet(dataSet, timeSteps, feature):
    # reshape & split dataSet
    dataX, dataY = [], []
    for i in range(0, len(dataSet) - timeSteps - 1, step=1):
        a = dataSet[i:(i + timeSteps), 0:feature]
        dataX.extend(a)
        dataY.append(dataSet[i + timeSteps - 1, feature])
    return np.array(dataX), np.array(dataY)

