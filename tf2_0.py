#%%
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist


gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

n_classes = 10
model = tf.keras.Sequential([
 tf.keras.layers.Conv2D(
 32, (5, 5), activation=tf.nn.relu, input_shape=(28, 28, 1)),
 tf.keras.layers.MaxPool2D((2, 2), (2, 2)),
 tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
 tf.keras.layers.MaxPool2D((2, 2), (2, 2)),
 tf.keras.layers.Flatten(),
 tf.keras.layers.Dense(1024, activation=tf.nn.relu),
 tf.keras.layers.Dropout(0.5),
 tf.keras.layers.Dense(n_classes)
])

model.summary()

(train_x, train_y), (test_x, test_y) = fashion_mnist.load_data()
# Scale input in [-1, 1] range
train_x = train_x / 255. * 2 - 1
test_x = test_x / 255. * 2 - 1
train_x = tf.expand_dims(train_x, -1).numpy()
test_x = tf.expand_dims(test_x, -1).numpy()

model.compile(
 optimizer=tf.keras.optimizers.Adam(1e-5),
 loss='sparse_categorical_crossentropy',
 metrics=['accuracy'])



#%%
model.fit(train_x, train_y, epochs=10)
model.evaluate(test_x, test_y)

#%%
import tensorflow as tf

A = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
x = tf.constant([[0, 10], [0, 0.5]])
b = tf.constant([[1, -1]], dtype=tf.float32)
y = tf.add(tf.matmul(A, x), b, name="result")
print(y)
print(y.numpy())
#%%
import tensorflow as tf

x = tf.Variable(1, dtype=tf.int32)
y = tf.Variable(2, dtype=tf.int32)

for _ in range(5):
    y.assign_add(1)
    out = x * y
    print(out)

#%%
import tensorflow as tf

@tf.function
def f():
    x = 0
    for i in range(10):
        print(i)
        x += i
    return x


f()
print(tf.autograph.to_code(f.python_function))

#%%
@tf.function
def output():
    for i in range(10):
        tf.print(i)


#%%
