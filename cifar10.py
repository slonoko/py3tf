# %% Learning CNN
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, regularizers, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    # .set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])

EPOCHS = 50
NUM_CLASSES = 10


def load_data():
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # normalize 
    mean = np.mean(x_train, axis=(0, 1, 2, 3))
    std = np.std(x_train, axis=(0, 1, 2, 3))
    x_train = (x_train - mean) / (std + 1e-7)
    x_test = (x_test - mean) / (std + 1e-7)

    y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)
    return x_train, y_train, x_test, y_test


(x_train, y_train, x_test, y_test) = load_data()


def build_model():
    model = models.Sequential()

    # 1st block
    model.add(layers.Conv2D(32, (3, 3), padding='same',
                            input_shape=x_train.shape[1:], activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.2))
    # 2nd block
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.3))
    # 3d block 
    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.4))
    # dense  
    model.add(layers.Flatten())
    model.add(layers.Dense(NUM_CLASSES, activation='softmax'))
    model.summary()
    return model


# %% Running ...

datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
)

# datagen.fit(x_train)

callbacks = [
    # Write TensorBoard logs to './logs' directory
    tf.keras.callbacks.TensorBoard(log_dir='logs')
]
model = build_model()
model.compile(loss='categorical_crossentropy',
              optimizer='RMSprop',
              metrics=['accuracy'])
# train

# model.fit_generator(datagen.flow(x_train, y_train, batch_size=64), callbacks=callbacks, epochs=EPOCHS, validation_data=(x_test, y_test), verbose=1)
model.fit(x_train, y_train, batch_size=64, callbacks=callbacks, epochs=EPOCHS, validation_data=(x_test, y_test),
          verbose=1)
# save to disk
model_json = model.to_json()
with open('cifar10_architecture.json', 'w') as json_file:
    json_file.write(model_json)
model.save_weights('cifar10_weights.h5')

score = model.evaluate(x_test, y_test, batch_size=128, verbose=1)

print("\nTest score:", score[0])
print('Test accuracy:', score[1])
print('Meaning, test result: %.3f loss: %.3f' % (score[1] * 100, score[0]))

# %%
