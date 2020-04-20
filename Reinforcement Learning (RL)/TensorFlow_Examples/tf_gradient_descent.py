# simple tensorflow example of using gradient descent

import numpy as np
import tensorflow as tf


A = tf.placeholder(tf.float32, shape=(5, 5), name='A')  # creates a matrix
v = tf.placeholder(tf.float32)  # creates a vector

w = tf.matmul(A, v)  # create tensor

# create TensorFlow execution session
with tf.Session() as session:
    output = session.run(w, feed_dict={A: np.random.randn(5, 5), v: np.random.randn(5, 1)})
    print(output, type(output))


# example of function optimization using gradient descent
u = tf.Variable(20.0)

cost = u*u + u + 1
lr = 0.3   # learning rate

train_op = tf.train.GradientDescentOptimizer(lr).minimize(cost)
init = tf.initialize_all_variables()

with tf.Session() as session:
    session.run(init)
    for i in range(32):
        session.run(train_op)
        print("i = %d, cost = %.3f, u = %.3f" % (i, cost.eval(), u.eval()))
