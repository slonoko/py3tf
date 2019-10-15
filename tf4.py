#%%
import numpy as np

np.random.seed(123)

from keras.models import Sequential
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

(X_train, y_train), (X_test, y_test) = mnist.load_data()


#%%
X_train = X_train.reshape(X_train.shape[0],1,28,28)
X_test = X_test.reshape(X_test.shape[0],1,28,28)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
print(X_train.shape)

#%%
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

num_classes = y_test.shape[1]
print(num_classes)
print(X_train.shape)

#%%
model = Sequential()
model.add(Conv2D(32,(5,5),input_shape=(1,28,28),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(240, activation='relu'))
model.add(Dense(num_classes,activation='softmax'))
print(model.output_shape)

model.compile(loss='binary_crossentropy', optimizer='adagrad', metrics=['accuracy'])

#%%
model.fit(X_train, y_train, batch_size=200, validation_data=(X_test, y_test), epochs=1)


#%%
scores = model.evaluate(X_test, y_test, verbose=0)
print('CNN error % .2f%%' % (100-scores[1]*100))


#%%
model_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)
model.save_weights('model.h5')


#%%
