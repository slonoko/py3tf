#%%
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers, models
# create the base pre-trained model

gpus = tf.config.experimental.list_physical_devices('GPU')

if len(gpus)>0:
    print("Using a GPU ...")
    tf.config.experimental.set_memory_growth(gpus[0], True)

base_model = InceptionV3(weights='imagenet', include_top=False)

# %%
x = base_model.output
# let's add a fully connected layer as first layer
x = layers.Dense(1024, activation='relu')(x)
# and a logistic layer with 200 classes as last layer
predictions = layers.Dense(200, activation='softmax')(x)
# model to train
model = models.Model(inputs=base_model.input, outputs=predictions)


# %%
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False


# %%
# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
# train the model on the new data for a few epochs
model.fit_generator(...)


# %%
