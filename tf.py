# %% TF version
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
print(tf.__version__)

#%%
g = tf.Graph()
t = np.random.rand(20240, 1)
r = np.random.rand(20240, 1)

%time f = np.dot(t, r.T)

with g.as_default():
    x = tf.placeholder(dtype=tf.float32, shape=(20240,1), name='x')
    y = tf.placeholder(dtype=tf.float32, shape=(20240,1), name='y')
    z = tf.matmul(x, y, transpose_b=True, name = 'z')
    init = tf.global_variables_initializer()

with tf.Session(graph=g) as sess:
    sess.run(init)
    %time v = sess.run(z, feed_dict={x: t, y: r})

#%%
import numpy as np
a = np.array([1, 2, 1])
w = np.array([[.5, .6], [.7, .8], [.7, .8]])

print(np.dot(a, w))
# [ 2.6  3. ] # plain nice old matix multiplication n x (n, m) -> m
print(np.sum(np.expand_dims(a, -1) * w , axis=0))
# equivalent result [2.6, 3]

import tensorflow as tf

a = tf.constant(a, dtype=tf.float64)
w = tf.constant(w)

with tf.Session() as sess:
  # they all produce the same result as numpy above
  print(tf.matmul(tf.expand_dims(a,0), w).eval())
  print((tf.reduce_sum(tf.multiply(tf.expand_dims(a,-1), w), axis=0)).eval())
  print((tf.reduce_sum(tf.multiply(a, tf.transpose(w)), axis=1)).eval())

  # Note tf.multiply is equivalent to "*"
  print((tf.reduce_sum(tf.expand_dims(a,-1) * w, axis=0)).eval())
  print((tf.reduce_sum(a * tf.transpose(w), axis=1)).eval())

# %%
# create a graph
g = tf.Graph()
with g.as_default():
    x = tf.placeholder(dtype=tf.float32,
                       shape=(None), name='x')
    w = tf.Variable(2.0, name='weight')
    b = tf.Variable(0.7, name='bias')

    z = w*x + b

    init = tf.global_variables_initializer()
# create a session and pass in graph g
with tf.Session(graph=g) as sess:
    # initialize w and b:
    sess.run(init)
    # evaluate z:
    for t in [1.0, 0.6, -1.8]:
        print('x=%4.1f --> z=%4.1f' % (
              t, sess.run(z, feed_dict={x: t})))


# %%

g = tf.Graph()
with g.as_default():
    x = tf.placeholder(dtype=tf.float32,
                       shape=(None, 2, 3),
                       name='input_x')

    x2 = tf.reshape(x, shape=(-1, 6),
                    name='x2')

    # calculate the sum of each column
    xsum = tf.reduce_sum(x2, axis=0, name='col_sum')

    # calculate the mean of each column
    xmean = tf.reduce_mean(x2, axis=0, name='col_mean')

with tf.Session(graph=g) as sess:
    x_array = np.arange(18).reshape(3, 2, 3)

    print('input shape: ', x_array.shape)
    print('Reshaped:\n',
          sess.run(x2, feed_dict={x: x_array}))
    print('Column Sums:\n',
          sess.run(xsum, feed_dict={x: x_array}))
    print('Column Means:\n',
          sess.run(xmean, feed_dict={x: x_array}))


# %%

X_train = np.arange(10).reshape((10, 1))
y_train = np.array([1.0, 1.3, 3.1,
                    2.0, 5.0, 6.3,
                    6.6, 7.4, 8.0,
                    9.0])


class TfLinreg(object):
    def __init__(self, x_dim, learning_rate=0.01,
                 random_seed=None):
        self.x_dim = x_dim
        self.learning_rate = learning_rate
        self.g = tf.Graph()
        # build the model
        with self.g.as_default():
            # set graph-level random-seed
            tf.set_random_seed(random_seed)

            self.build()
            # create initializer
            self.init_op = tf.global_variables_initializer()

    def build(self):
        # define placeholders for inputs
        self.X = tf.placeholder(dtype=tf.float32,
                                shape=(None, self.x_dim),
                                name='x_input')
        self.y = tf.placeholder(dtype=tf.float32,
                                shape=(None),
                                name='y_input')
        print(self.X)
        print(self.y)
        # define weight matrix and bias vector
        w = tf.Variable(tf.zeros(shape=(1)),
                        name='weight')
        b = tf.Variable(tf.zeros(shape=(1)),
                        name="bias")
        print(w)
        print(b)

        self.z_net = tf.squeeze(w*self.X + b,
                                name='z_net')
        print(self.z_net)

        sqr_errors = tf.square(self.y - self.z_net,
                               name='sqr_errors')
        print(sqr_errors)
        self.mean_cost = tf.reduce_mean(sqr_errors,
                                        name='mean_cost')

        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=self.learning_rate,
            name='GradientDescent')
        self.optimizer = optimizer.minimize(self.mean_cost)


# %%
lrmodel = TfLinreg(x_dim=X_train.shape[1], learning_rate=0.01)


def train_linreg(sess, model, X_train, y_train, num_epochs=10):
    # initialiaze all variables: W and b
    sess.run(model.init_op)

    training_costs = []
    for i in range(num_epochs):
        _, cost = sess.run([model.optimizer, model.mean_cost],
                           feed_dict={model.X: X_train,
                                      model.y: y_train})
        training_costs.append(cost)

    return training_costs


sess = tf.Session(graph=lrmodel.g)
training_costs = train_linreg(sess, lrmodel, X_train, y_train)

plt.plot(range(1, len(training_costs) + 1), training_costs)
plt.tight_layout()
plt.xlabel('Epoch')
plt.ylabel('Training Cost')
plt.show()


def predict_linreg(sess, model, X_test):
    y_pred = sess.run(model.z_net,
                      feed_dict={model.X: X_test})
    return y_pred


plt.scatter(X_train, y_train,
            marker='s', s=50,
            label='Training Data')
plt.plot(range(X_train.shape[0]),
         predict_linreg(sess, lrmodel, X_train),
         color='gray', marker='o',
         markersize=6, linewidth=3,
         label='LinReg Model')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.tight_layout()
plt.show()

# %%
