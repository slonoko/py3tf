#%%
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV, LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation

#%%
iris = load_iris()
X,y = iris.data[:, :4], iris.target

#%%
train_X, test_X, train_y, test_y = train_test_split(X,y,train_size=0.5, random_state=0)

#%%
model = Sequential()

model.add(Dense(16, input_shape=(4,)))
model.add(Activation('sigmoid'))

model.add(Dense(3))
model.add(Activation('softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


#%%
model.fit(train_X, train_y, verbose=1, batch_size=1, nb_epoch=100)


#%%
loss, accuracy = model.evaluate(test_X, test_y, verbose=0)
print(f'Accuracy is using keras prediction {accuracy}')

#%%
