# %%
import numpy as np
import tensorflow as tf

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
