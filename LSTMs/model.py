from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers import LSTM
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
        self.batch_size = 1
        self.layer_neurons = [10, 20]
        self.dropout = 0
        self.response_type = 'reg'

        for key, value in kwargs.items():
            setattr(self, key, value)
        print(self.layer_neurons)
        self.model = self.createLSTMs()

    def createLSTMs(self):
        tf.reset_default_graph()
        model = Sequential()

        for key in range(len(self.layer_neurons)):
            if key == 0:
                model.add(
                    LSTM(self.layer_neurons[key], input_shape=(self.timeSteps, self.feature), return_sequences=True))
            else:
                model.add(LSTM(self.layer_neurons[key], return_sequences=True))

        if self.dropout != 0:
            model.add(Dropout(self.dropout))
        model.add(Flatten())
        model.add(Dense(1, activation='softsign'))

        if self.response_type.upper() == 'CLASSIFICATION':
            print('Building classification model...')
            model.compile(loss='categorical_crossentropy', optimizer='RMSProp', metrics=['accuracy'])
            return model
        elif self.response_type.upper() == 'REG':
            print('Building regression model...')
            model.compile(loss='mean_squared_error', optimizer='RMSProp')
            return model
        else:
            print('> error: wrong response_type')

    def get_model(self):
        return self.model

    def save_model_weight(self, filename):
        # save model para to file
        net_para_file = 'LSTMs/Model_Weight_Data/' + filename + '.h5'
        self.model.save_weights(net_para_file)

    def load_model_weight(self, filename):
        # load model para fro  file
        net_para_file = 'LSTMs/Model_Weight_Data/' + filename + '.h5'
        if os.path.exists(net_para_file):
            self.model.load_weights(net_para_file)
        else:
            print('> error: ' + filename + ' is not found')

    def train(self, trainX, trainY, epochs=1, batch=1):
        for i in range(epochs):
            print(i+1)
            self.model.fit(trainX, trainY, epochs=1, batch_input_shape=(batch, self.timeSteps, self.feature), shuffle=False)
            self.model.reset_states()


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
