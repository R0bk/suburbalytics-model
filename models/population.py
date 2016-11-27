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


def load_csv(file, row=-1):
    csvd = '../data-collection/population/' + file

    data = np.genfromtxt(csvd, delimiter=',')
    if row != -1:
        X = data[row, 1:]
        y = [X[i - 1] for i, x in enumerate(X) if i != 0]
    else:
        X = data[:, 0]
        y = data[:, 1]
        return(X, y)
    raw_X, raw_Y = X, y
    X = scaler.fit_transform(X)
    y = scaler.fit_transform(y)
    norm_X = normalize(X)
    split = int(0.7 * len(X))
    return(X, y, split, data, norm_X, raw_X, raw_Y)


def list_of_lists(array):
    return(np.array([[x] for x in array]))


def model(file, n, row, wr):
    X, y, split, data, norm_X, raw_X, raw_Y = load_csv(file, row)

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
    model.fit(trainX, trainY, nb_epoch=150, batch_size=1, verbose=2)
    print(trainX)
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform(trainY)
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform(testY)

    future = []
    for i in range(n):
        if not len(future):
            value = scaler.inverse_transform(model.predict(testX))[-1]
        else:
            print('**********************************************')
            scaled = scaler.fit_transform(list_of_lists(np.append(raw_X, future)))
            arranged = np.reshape(scaled, (scaled.shape[0], 1, scaled.shape[1]))
            print(arranged)
            value = scaler.inverse_transform(model.predict(arranged))[-1]
        future.append(value)
    with open('future' + file, 'a') as f:
        wr = csv.writer(f, delimiter=',')
        wr.writerow(future)


def predicter(n, data):

    for file in os.listdir(data):
        with open('future' + file, 'w') as f:
            wr = csv.writer(f, delimiter=',')
        print('processing:\t' + file)
        for row in range(len(load_csv(file)[0])):
            print('processing row:\t' + str(row))
            model(file, n, row, wr)


predicter(10, '../data-collection/population/')
