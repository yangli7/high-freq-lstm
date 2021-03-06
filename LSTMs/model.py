from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers import LSTM
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import numpy as np
import os


class LSTMs:
    """
    Inputs:
        - timeSteps: number of time sequence inputs
        - feature: number of sample feature
        - layer_neurons: number of LSTMs neurons in each layer
        - dropout: dropout rate
        - response_type: 'classification' or 'regression'
    """

    def __init__(self, **kwargs):
        self.timeSteps = 5
        self.feature = 1
        self.layer_neurons = [10, 20]
        self.dropout = 0
        self.batch_size = 1
        self.response_type = 'reg'

        for key, value in kwargs.items():
            setattr(self, key, value)
        self.model = self.createLSTMs()

    def createLSTMs(self):
        tf.reset_default_graph()
        model = Sequential()
        for key in range(len(self.layer_neurons)):
            if key == 0:
                model.add(
                    LSTM(self.layer_neurons[key], input_shape=(self.timeSteps, self.feature),
                         return_sequences=True))
                if self.dropout != 0:
                    model.add(Dropout(self.dropout))
            else:
                model.add(
                    LSTM(self.layer_neurons[key], return_sequences=True))
                if self.dropout != 0:
                    model.add(Dropout(self.dropout))
        model.add(Flatten())
        if self.response_type.upper() == 'CLASSIFICATION':
            print('Building classification model...')
            model.add(Dense(1, activation='softmax'))
            model.compile(loss='categorical_crossentropy', optimizer='RMSProp', metrics=['accuracy'])
        elif self.response_type.upper() == 'REG':
            print('Building regression model...')
            model.add(Dense(1, activation='linear'))
            model.compile(loss='mean_squared_error', optimizer='RMSProp')
        else:
            print('> error: wrong response_type')
        print(model.summary())
        return model

    def get_model(self):
        return self.model

    def save_model_weight(self, filename):
        # save model para to file
        net_para_file = 'Model_Weight_Data/' + filename + '.h5'
        self.model.save_weights(net_para_file)

    def load_model_weight(self, filename):
        # load model para fro  file
        net_para_file = 'Model_Weight_Data/' + filename + '.h5'
        if os.path.exists(net_para_file):
            self.model.load_weights(net_para_file)
        else:
            print('> error: ' + filename + ' is not found')

    def train(self, trainX, trainY, epochs=1):
        for i in range(epochs):
            print(i + 1)
            self.model.fit(trainX, trainY, epochs=1, batch_size=self.batch_size, verbose=1)
            self.model.reset_states()

    def predict(self, testX, reset_stateful=0):
        trainPredict = self.model.predict(testX, batch_size=self.batch_size)
        if reset_stateful == 1:
            self.model.reset_states()
        return trainPredict

    def import_model(self, model):
        self.model = model


# encode for classification
def one_hot_encode(trainY, n_label):
    encoding = list()
    for value in trainY:
        vector = [0 for _ in range(n_label)]
        vector[value] = 1
        encoding.append(vector)
    return np.array(encoding)


# decode for encoded sequence
def one_hot_decode(encoded_seq):
    return [np.argmax(vector) for vector in encoded_seq]


# transform raw_data as a window dataSet
def data_trans(raw_data, time_step):
    data = np.array(raw_data)
    window_num = data.shape[0] - time_step + 1
    dataX = []
    dataY = []
    for i in range(window_num):
        dataX.append(data[i:time_step + i, 0:data.shape[1] - 1])
        dataY.append(data[time_step + i - 1, -1])
    return np.array(dataX), np.array(dataY)


# normalize dataSet
def normalize(dataSet):
    scaler = StandardScaler()
    dataSet = scaler.fit_transform(dataSet)
    return dataSet, scaler


# invert dataSet
def inverse(dataSet, scaler):
    return scaler.inverse_transform(dataSet)
