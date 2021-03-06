# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os
import numpy as np
import re
import shutil
import tensorflow as tf
DATA_DIR = ".data"
CHECKPOINT_DIR = os.path.join(DATA_DIR, "checkpoints")

# %%

y_true = [1, 2]
y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
assert loss.shape == (2,)
loss.numpy()

# %%
def download_and_read(urls):
   texts = []
   for i, url in enumerate(urls):
       p = tf.keras.utils.get_file("ex1-{:d}.txt".format(i), url,
           cache_dir=".")
       text = open(p, "r").read()
       # remove byte order mark
       text = text.replace("\ufeff", "")
       # remove newlines
       text = text.replace('\n', ' ')
       text = re.sub(r'\s+', " ", text)
       # add it to the list
       texts.extend(text)
   return texts
texts = download_and_read([
   "http://www.gutenberg.org/cache/epub/28885/pg28885.txt",
   "https://www.gutenberg.org/files/12/12-0.txt"
])


# %%
# create the vocabulary
vocab = sorted(set(texts))
print("vocab size: {:d}".format(len(vocab)))
# create mapping from vocab chars to ints
char2idx = {c:i for i, c in enumerate(vocab)}
idx2char = {i:c for c, i in char2idx.items()}


# %%
# numericize the texts
texts_as_ints = np.array([char2idx[c] for c in texts])
data = tf.data.Dataset.from_tensor_slices(texts_as_ints)
# number of characters to show before asking for prediction
# sequences: [None, 100]
seq_length = 100
sequences = data.batch(seq_length + 1, drop_remainder=True)
def split_train_labels(sequence):
   input_seq = sequence[0:-1]
   output_seq = sequence[1:]
   return input_seq, output_seq
sequences = sequences.map(split_train_labels)
# set up for training
# batches: [None, 64, 100]
batch_size = 64
steps_per_epoch = len(texts) // seq_length // batch_size
dataset = sequences.shuffle(10000).batch(
    batch_size, drop_remainder=True)


# %%
class CharGenModel(tf.keras.Model):
   def __init__(self, vocab_size, num_timesteps,
           embedding_dim, **kwargs):
       super(CharGenModel, self).__init__(**kwargs)
       self.embedding_layer = tf.keras.layers.Embedding(
           vocab_size,
           embedding_dim
       )
       self.rnn_layer = tf.keras.layers.GRU(
           num_timesteps,
           recurrent_initializer="glorot_uniform",
           recurrent_activation="sigmoid",
           stateful=True,
           return_sequences=True)
       self.dense_layer = tf.keras.layers.Dense(vocab_size)
   def call(self, x):
       x = self.embedding_layer(x)
       x = self.rnn_layer(x)
       x = self.dense_layer(x)
       return x
vocab_size = len(vocab)
embedding_dim = 256

model = CharGenModel(vocab_size, seq_length, embedding_dim)
model.build(input_shape=(batch_size, seq_length))


# %%
def loss(labels, predictions):
   return tf.losses.sparse_categorical_crossentropy(
       labels,
       predictions,
       from_logits=True
   )
model.compile(optimizer=tf.optimizers.Adam(), loss=loss)


# %%
def generate_text(model, prefix_string, char2idx, idx2char,
       num_chars_to_generate=1000, temperature=1.0):
   input = [char2idx[s] for s in prefix_string]
   input = tf.expand_dims(input, 0)
   text_generated = []
   model.reset_states()
   for i in range(num_chars_to_generate):
       preds = model(input)
       preds = tf.squeeze(preds, 0) / temperature
       # predict char returned by model
       pred_id = tf.random.categorical(
           preds, num_samples=1)[-1, 0].numpy()
       text_generated.append(idx2char[pred_id])
       # pass the prediction as the next input to the model
       input = tf.expand_dims([pred_id], 0)
   return prefix_string + "".join(text_generated)


# %%
num_epochs = 50
for i in range(num_epochs // 10):
   model.fit(
       dataset.repeat(),
       epochs=10,
       steps_per_epoch=steps_per_epoch
       # callbacks=[checkpoint_callback, tensorboard_callback]
   )
   checkpoint_file = os.path.join(
       CHECKPOINT_DIR, "model_epoch_{:d}".format(i+1))
   model.save_weights(checkpoint_file)
   # create generative model using the trained model so far
   gen_model = CharGenModel(vocab_size, seq_length, embedding_dim)
   gen_model.load_weights(checkpoint_file)
   gen_model.build(input_shape=(1, seq_length))
   print("after epoch: {:d}".format(i+1)*10)
   print(generate_text(gen_model, "Alice ", char2idx, idx2char))
   print("---")


# %%


