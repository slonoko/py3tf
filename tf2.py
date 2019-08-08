#%%
import tensorflow as tf

my_graph = tf.Graph()

with my_graph.as_default():
    x = tf.placeholder('float32', shape=(None, 4))
    y = 10*x+5
    init = tf.local_variables_initializer()

with tf.Session(graph=my_graph) as sess:
    sess.run(init)

    x_dict = [[1,2,5,6], [8,5,3,7]]

    print(sess.run(y, feed_dict={x:x_dict}))


#%%
image = tf.image.decode_png(tf.read_file('image.png'), channels=3)
img_sess = tf.InteractiveSession()
print(img_sess.run(tf.shape(image)))

#%%
r1 = tf.random_uniform([2,3], minval=0, maxval=5)
print(img_sess.run(r1))

#%%
img_sess.close()

#%%
opt_graph = tf.Graph()

with opt_graph.as_default():
    x = tf.Variable(3, name = 'x', dtype = 'float32')
    log_x = tf.log(x)
    log_x_squared = tf.square(log_x)
    optimizer = tf.train.GradientDescentOptimizer(0.7)
    train = optimizer.minimize(log_x_squared)
    init  = tf.global_variables_initializer()

with tf.Session(graph=opt_graph) as sess2:
    sess2.run(init)
    print(f'starting at x: {sess2.run(x)}, log(x)^2: {sess2.run(log_x_squared)}')
    for step in range(20):
        sess2.run(train)
        print(f'step {step}, with x: {sess2.run(x)}, log(x)^2: {sess2.run(log_x_squared)}')

#%%
import numpy as np
import os
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam, SGD
from keras.utils import np_utils

np.random.seed(100)
(x_train, y_train), (x_test, y_test) = cifar10.load_data()



#%%
x_train = x_train.reshape(50000, 3072)
x_test = x_test.reshape(10000,3072)

x_train = (x_train - np.mean(x_train))/np.std(x_train)
x_test = (x_test - np.mean(x_test))/np.std(x_test)

#%%
labels = 10
y_train = np_utils.to_categorical(y_train, labels)
y_test = np_utils.to_categorical(y_test, labels)

#%%
model = Sequential()
model.add(Dense(512, input_shape=(3072,)))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(Dense(120))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(labels))
model.add(Activation('sigmoid'))

#%%
adam = Adam(0.01)
sgd = SGD(0.01)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
#%%
model.fit(x_train, y_train, batch_size=1000, nb_epoch=50, validation_data=(x_test,y_test))
#%%
score = model.evaluate(x_test, y_test, verbose=0)
print(f'Test accuracy is {score[1]}')

#%%
model.predict_classes(x_test)

#%%
model.save('model.h5')
jsonModel = model.to_json()
model.save_weights('model_weights.h5')

#%%
model.summary()

#%%
from keras.callbacks import EarlyStopping
early_stopping_monitor = EarlyStopping(patience=2)
model.fit(x_train, y_train, batch_size=1000, nb_epoch=10, validation_data=(x_test,y_test), callbacks=[early_stopping_monitor])
#%%
model.output_shape
#%%
