import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import boston_housing

(XTrain, YTrain),(XTest, YTest) = boston_housing.load_data()

myANN = Sequential()
myANN.add(Dense(10,input_dim=13,kernel_initializer="normal",activation="relu"))
myANN.add(Dense(10,kernel_initializer="random_uniform",activation="relu"))
myANN.add(Dense(1,kernel_initializer="normal",activation="linear"))
myANN.compile(loss="mean_squared_error",optimizer="adam")
myANN.fit(XTrain,YTrain, epochs=1000, verbose=True)

yP = myANN.predict(XTest)
errorT = np.abs(yP-YTest)
meanE = np.mean(np.abs(errorT))
maxE = np.max(np.abs(errorT))
minE = np.min(np.abs(errorT))
print("Mean: %e, Max: %e, Min: %e" % (meanE,maxE,minE))
