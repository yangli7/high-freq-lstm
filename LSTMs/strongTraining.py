from keras.layers import LSTM

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from LSTMs import model

from keras.models import Sequential
from keras.layers.core import Dense, Flatten
from keras.layers import LSTM

'''
feature list
'buy3', 'buy4', 'buy5', 'bc3', 'bc4', 'bc5', 'sale3', 'sale4', 'sale5', 'sc3', 'sc4', 'sc5'
'buy1', 'buy2', 'bc1', 'bc2',
                     'sale1', 'sale2', 'sc1', 'sc2',
'''

# parameter
filename = 'database_SH600637.csv'
feature_label_name = ['price', 'buy1', 'buy2', 'bc1', 'bc2', 'sale1', 'sale2', 'sc1', 'sc2', '5min']
feature = len(feature_label_name)-1
train_rate = 0.67

time_step = 30
batch_size = 150
layer_neurons = [20, 10]
dropout = 0.4

if_train = 1
epochs = 100
modelname = 'model_test'


# create trainSet & testSet
dataframe = pd.read_csv('database/'+filename, encoding='gbk')
dataset = dataframe[feature_label_name].values

dataset[:, 0:-1], scaler_Feature = model.normalize(dataset[:, 0:-1])
label = dataset[:, -1]
label = np.reshape(label, (len(label), 1))
label, scaler_Label = model.normalize(label)
label = np.reshape(label, (len(label)))
dataset[:, -1] = label
train_size = int(len(dataset) * train_rate)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
trainX, trainY = model.data_trans(train, time_step)
testX, testY = model.data_trans(test, time_step)


# create model
LSTM_ = model.LSTMs(timeSteps=time_step, feature=feature, batch_size=batch_size, layer_neurons=layer_neurons, dropout=dropout)

# train or load
if if_train == 1:
    LSTM_.train(trainX, trainY, epochs)
    LSTM_.save_model_weight(modelname)
elif if_train == 2:
    model_ = Sequential()
    model_.add(LSTM(10, input_shape=(time_step, feature), return_sequences=True))
    model_.add(LSTM(20, return_sequences=True))
    model_.add(Flatten())
    model_.add(Dense(1))
    model_.compile(loss='mean_squared_error', optimizer='adam')
    print(model_.summary())
    model_.fit(trainX, trainY, epochs=4, shuffle=False)
    LSTM_.import_model(model_)
else:
    LSTM_.load_model_weight(modelname)
# predict
predict_train = LSTM_.predict(trainX)
predict_test = LSTM_.predict(testX)


# inverse
trainY = np.reshape(trainY, (trainY.shape[0], 1))
trainY = model.inverse(trainY, scaler_Label)
predict_train = np.reshape(predict_train, (predict_train.shape[0], 1))
predict_train = model.inverse(predict_train, scaler_Label)
testY = np.reshape(testY, (testY.shape[0], 1))
testY = model.inverse(testY, scaler_Label)
predict_test = np.reshape(predict_test, (predict_test.shape[0], 1))
predict_test = model.inverse(predict_test, scaler_Label)

trainY = np.reshape(trainY, (trainY.shape[0]))
predict_train = np.reshape(predict_train, (predict_train.shape[0]))
testY = np.reshape(testY, (testY.shape[0]))
predict_test = np.reshape(predict_test, (predict_test.shape[0]))


plt.figure(figsize=(50, 15))
plt.plot(trainY)
plt.plot(predict_train)
plt.show()



