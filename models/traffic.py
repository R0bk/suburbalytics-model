from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import keras
import csv
import sys
import os

csvd = '../data/data-collection/traffic/data.csv'


def load_csv():
    data = np.genfromtxt(csvd, delimiter=',')
    X = data[1:15, 1]
    y = data[:]

    print(X)

    norm_X = preprocessing.normalize(X)
    stand_X = preprocessing.scale(X)
    split = int(0.7 * len(X))
    return(X, y, split, data, norm_X, stand_X)

X, y, split, data, norm_X, stand_X = load_csv()
test_size = len(norm_X) - split
train, test = norm_X[0: split, :], norm_X[split: len(norm_X), :]

look_back = 1
trainX, trainY = create_norm_X(train, look_back)
testX, testY = create_norm_X(test, look_back)

trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

model = Sequential()
model.add(LSTM(4, input_dim=look_back))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, nb_epoch=100, batch_size=1, verbose=2)

trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
print('Test Score: %.2f RMSE' % (testScore))

trainPredictPlot = numpy.empty_like(norm_X)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict

testPredictPlot = numpy.empty_like(norm_X)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict) + (look_back * 2) +
                1:len(norm_X) - 1, :] = testPredict

plt.plot(scaler.inverse_transform(norm_X))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()
