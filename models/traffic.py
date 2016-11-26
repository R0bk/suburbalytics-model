from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler, normalize, scale
from sklearn.metrics import mean_squared_error
import numpy as np
import keras
import math
import csv
import sys
import os

scaler = MinMaxScaler(feature_range=(0, 1))


def load_csv(file):
    csvd = '../data-collection/traffic/' + file
    data = np.genfromtxt(csvd, delimiter=',')
    X = data[1:15, 0]
    y = data[1:15, 1]
    
    X = scaler.fit_transform(X)
    y = scaler.fit_transform(y)
    norm_X = normalize(X)
    stand_X = scale(X)
    split = int(0.7 * len(X))
    return(X, y, split, data, norm_X, stand_X)


def list_of_lists(array):
    return(np.array([[x] for x in array]))


def model(file, n): 
    X, y, split, data, norm_X, stand_X = load_csv(file)

    test_size = len(norm_X) - split
    X = np.array([x for x in X])
    y = np.array([x for x in y])
    #train, test = norm_X[0: split, :], norm_X[split: len(norm_X), :]

    look_back = 1
    trainX, trainY = (list_of_lists(X[:split]), list_of_lists(y[:split]))
    testX, testY = (list_of_lists(X[split:]), list_of_lists(y[split:]))

    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, trainX.shape[1]))

    model = Sequential()
    model.add(LSTM(4, input_dim=look_back))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, nb_epoch=100, batch_size=1, verbose=2)

    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform(trainY)
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform(testY)
    print(testPredict)

    future = []
    for i in range(n):
        if not len(future):
            value = model.predict(testX)[-1]
        else:
            print(list_of_lists(future))
            value = model.predict(list_of_lists(future))[-1]
        future.append(value)
    with open('future' + file, 'w') as f:
        wr = csv.writer(f, delimiter=',')
        wr.writerow(future)

def predicter(n, data):
    for file in os.listdir(data):
    	print('processing:\t' + file)
        model(file, n)

predicter(12, '../data-collection/traffic/')