import tensorflow as tf
import os
import numpy as np
from tensorflow.keras import datasets, layers, models, preprocessing
import tensorflow_datasets as tfds

max_len = 500
n_words = 88000
dim_embedding = 32
EPOCHS = 2
BATCH_SIZE = 128
model_dir = '.data/imdb'

pos_review = "Joaquin Phoenix gives a tour de force performance, fearless and stunning in its emotional depth and " \
            "physicality. It's impossible to talk about this without referencing Heath Ledger's Oscar-winning " \
            "performance from The Dark Knight, widely considered the definitive live-action portrayal of the Joker, " \
            "so let's talk about it. The fact is, everyone is going to be stunned by what Phoenix accomplishes, " \
            "because it's what many thought impossible - a portrayal that matches and potentially exceeds that of The " \
            "Dark Knight's Clown Prince of Crime "

neg_review = "Once again Mr. Costner has dragged out a movie for far longer than necessary. Aside from the terrific " \
             "sea rescue sequences, of which there are very few I just did not care about any of the characters. Most " \
             "of us have ghosts in the closet, and Costner's character are realized early on, and then forgotten " \
             "until much later, by which time I did not care. The character we should really care about is a very " \
             "cocky, overconfident Ashton Kutcher. The problem is he comes off as kid who thinks he's better than " \
             "anyone else around him and shows no signs of a cluttered closet. His only obstacle appears to be " \
             "winning over Costner. Finally when we are well past the half way point of this stinker, Costner tells " \
             "us all about Kutcher's ghosts. We are told why Kutcher is driven to be the best with no prior inkling " \
             "or foreshadowing. No magic here, it was all I could do to keep from turning it off an hour in. "

gpus = tf.config.experimental.list_physical_devices("GPU")

if len(gpus) > 0:
    print("Using a GPU ...")
    tf.config.experimental.set_memory_growth(gpus[0], True)


def prepare_embedding(review):
    encoded_doc = tf.keras.preprocessing.text.one_hot(review, n_words)
    padded_doc = preprocessing.sequence.pad_sequences([encoded_doc], maxlen=max_len, padding="post")
    return padded_doc


def load_data():
    # load data
    (X_train, y_train), (X_test, y_test) = datasets.imdb.load_data(num_words=n_words)

    X = np.concatenate((X_train, X_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)
    print("Training data: ")
    print(X.shape)
    print(y.shape)
    print("Classes: ")
    print(np.unique(y))
    print("Number of words: ")
    print(len(np.unique(np.hstack(X))))
    print("Review length: ")
    result = [len(x) for x in X]
    print("Mean %.2f words (%f)" % (np.mean(result), np.std(result)))

    # Pad sequences with max_len
    X_train = preprocessing.sequence.pad_sequences(X_train, maxlen=max_len)
    X_test = preprocessing.sequence.pad_sequences(X_test, maxlen=max_len)
    return (X_train, y_train), (X_test, y_test)


def build_model():

    model = models.Sequential()
    """
    model.add(layers.Embedding(n_words, dim_embedding, input_length=max_len))
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv1D(256, 3, padding="valid", activation="relu"))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation="sigmoid"))
    """
    model.add(layers.Embedding(n_words, dim_embedding, input_length=max_len))
    model.add(layers.Flatten())
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


(X_train, y_train), (X_test, y_test) = load_data()
model = build_model()
model.summary()

callbacks = [tf.keras.callbacks.TensorBoard(log_dir=f'{model_dir}/logs'),
             tf.keras.callbacks.ModelCheckpoint(f'{model_dir}/model/weights', save_weights_only=True)]

model.compile(optimizer=tf.optimizers.Adam(), loss=tf.losses.BinaryCrossentropy(), metrics=["accuracy"])

if os.path.exists(model_dir):
    model.load_weights(f'{model_dir}/model/weights')
else:
    model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
#        callbacks=callbacks
    )
    scores = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
    print("Accuracy: %.2f%%" % (scores[1] * 100))

model.save_weights(f'{model_dir}/model/weights')

pos_result = model.predict(prepare_embedding(pos_review))
neg_result = model.predict(prepare_embedding(neg_review))
print(pos_result, neg_result)
