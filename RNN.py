import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import random

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, GRU, RNN
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

data = pd.read_csv('PWM vs RPM_18-05.csv')
# The data needs to be in csv format and in a folder named 'Intern_dataset'

print(data.shape)
dataset = data.values.astype('float32')
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)


# Accessing the columns independently
motor_pwm = dataset[:, 0]
rpm = dataset[:, 1]


# Knowing where the Motor Voltage i.e. is zero for better data handling
zeros_indices = np.where(motor_pwm == 0)
print(zeros_indices)

t = zeros_indices[0]

# Tells the length of each cycle
r = [j - i for i, j in zip(t[:-1], t[1:])]


# Initializing dictionaries for input and output
d_i = {}
d_o = {}

# Values are 1-D numpy arrays for separate cycle
for x in range(len(r)):
    d_i['input {0}'.format(x + 1)] = motor_pwm[t[x]:t[x + 1]]
    d_o['output {0}'.format(x + 1)] = rpm[t[x]:t[x + 1]]



# Splitting of dataset
testX = np.array([])
testY = np.array([])
trainX = np.array([])
trainY = np.array([])

i_choices = []
o_choices = []


# Length of Test data
while len(testX) < 1e4:
    ikey = random.choice(list(d_i))
    if ikey in i_choices:
        continue
    else:
        i_choices.append(ikey)
        testX = np.concatenate([testX, d_i[ikey]])
        okey = 'output ' + [int(s) for s in ikey.split() if s.isdigit()][0].__str__()
        o_choices.append(okey)
        testY = np.concatenate([testY, d_o[okey]])

Xkey = list(d_i)
Ykey = list(d_o)

for i in i_choices:
    Xkey.remove(i)

for o in o_choices:
    Ykey.remove(o)

for x in Xkey:
    trainX = np.concatenate([trainX, d_i[x]])

for y in Ykey:
    trainY = np.concatenate([trainY, d_o[y]])


def create_dataset(X, Y, look_back=1):
    dataX, dataY = [], []
    for i in range(len(X) - look_back - 1):
        a = X[i:(i + look_back)]
        dataX.append(a)
        dataY.append(Y[i + look_back])
    return np.array(dataX), np.array(dataY)


# Creating the data shape for training RNN with timepteps dependency in look_back
look_back = 16


Xtrain, Ytrain = create_dataset(trainX, trainY, look_back=look_back)
Xtest, Ytest = create_dataset(testX, testY, look_back=look_back)


X_train = np.reshape(Xtrain, (Xtrain.shape[0], Xtrain.shape[1], 1))
X_test = np.reshape(Xtest, (Xtest.shape[0], Xtest.shape[1], 1))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(8, input_shape=(look_back, 1)))
model.add(Dense(1, activation='relu'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'mae'])
model.summary()
model.fit(X_train, Ytrain, epochs=100, batch_size=32, verbose=2, validation_data=(X_test, Ytest))

# Evaluating the model
scores = model.evaluate(X_test, Ytest)
print(model.metrics_names)
print(scores)

# Predicting
trainPredict = model.predict(X_train)
testPredict = model.predict(X_test)

# inverting the scalar
ytrain = np.zeros(shape=(len(trainPredict), 2))
ytrain[:, 0] = trainPredict[:, 0]
ytrain[:, 1] = Ytrain
ytrain = scaler.inverse_transform(ytrain)

ytest = np.zeros(shape=(len(testPredict), 2))
ytest[:, 0] = testPredict[:, 0]
ytest[:, 1] = Ytest
ytest = scaler.inverse_transform(ytest)

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(ytrain[:, 0], ytrain[:, 1]))
print('Train Score: %.2f RMSE' % trainScore)
testScore = math.sqrt(mean_squared_error(ytest[:, 0], ytest[:, 1]))
print('Train Score: %.2f RMSE' % testScore)
