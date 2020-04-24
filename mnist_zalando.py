import os
import time
import tensorflow as tf
import numpy as np
from tensorflow.keras import models, layers

gpus = tf.config.experimental.list_physical_devices('GPU')

if len(gpus)>0:
    print("Using a GPU ...")
    tf.config.experimental.set_memory_growth(gpus[0], True)

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

TRAINING_SIZE = len(train_images)
TEST_SIZE = len(test_images)
LABEL_DIMENSION = 10
OPTIMIZER = tf.keras.optimizers.SGD()

train_images = np.asarray(train_images, dtype=np.float32) / 255.0
test_images = np.asarray(test_images, dtype=np.float32) / 255.0

train_images = train_images.reshape((TRAINING_SIZE, 28,28, 1))
test_images = test_images.reshape((TEST_SIZE, 28,28,1))

train_labels = tf.keras.utils.to_categorical(train_labels, LABEL_DIMENSION).astype(np.float32)
test_labels = tf.keras.utils.to_categorical(test_labels, LABEL_DIMENSION).astype(np.float32)

model = models.Sequential()
model.add(layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)))
model.add(layers.MaxPooling2D(pool_size=(2,2), strides=2))
model.add(layers.Conv2D(filters=64,kernel_size=(3,3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2), strides=2))
model.add(layers.Conv2D(filters=64,kernel_size=(3,3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(LABEL_DIMENSION,activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])

#strategy = None
strategy = tf.distribute.MirroredStrategy()
config = tf.estimator.RunConfig(train_distribute=strategy)

estimator = tf.keras.estimator.model_to_estimator(model, config=config)


def input_fn(images, labels, epochs, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    SHUFFLE_SIZE = 5000
    dataset = dataset.shuffle(SHUFFLE_SIZE).repeat(epochs).batch(batch_size)
    dataset = dataset.prefetch(None)
    return dataset

BATCH_SIZE = 512
EPOCHS = 50

estimator_train_result = estimator.train(input_fn=lambda:input_fn(train_images, train_labels, EPOCHS, BATCH_SIZE))
print(estimator_train_result)

score = estimator.evaluate(lambda:input_fn(test_images, test_labels,1, BATCH_SIZE))

print(f'The Score is {score}')