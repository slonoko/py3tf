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
    for step in range(10):
        sess2.run(train)
        print(f'step {step}, with x: {sess2.run(x)}, log(x)^2: {sess2.run(log_x_squared)}')

#%%
