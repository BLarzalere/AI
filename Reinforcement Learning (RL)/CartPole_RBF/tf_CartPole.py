# CartPole with TensorFlow

from __future__ import print_function, division

import numpy as np
import tensorflow as tf
import CartPole_RBF_Network


class SGDRegressor:
    def __init__(self, d):
        print("Hello TensorFlow")
        lr = 0.1

        # create inputs, targets and params
        # matmul doesn't like 1-D tensors, so we'll make w a 2-D matrix
        # and then flatten the prediction to make it 1-D
        self.w = tf.Variable(tf.random_normal(shape=(d, 1)), name='w')
        self.x = tf.placeholder(tf.float32, shape=(None, d), name='x')
        self.y = tf.placeholder(tf.float32, shape=(None, ), name='y')

        # make prediction and determine cost fnc value
        y_hat = tf.reshape( tf.matmul(self.x, self.w), [-1] )
        delta = self.y - y_hat
        cost = tf.reduce_sum(delta * delta)

        # ops we want to call later
        self.train_op = tf.train.GradientDescentOptimizer(lr).minimize(cost)
        self.predict_op = y_hat

        # start the tf session and initialize params
        init = tf.global_variables_initializer()
        self.session = tf.InteractiveSession()
        self.session.run(init)

    def partial_fit(self, x, y):
        self.session.run(self.train_op, feed_dict={self.x: x, self.y: y})

    def predict(self, x):
        return self.session.run(self.predict_op, feed_dict={self.x: x})


if __name__ == '__main__':
    CartPole_RBF_Network.SGDRegressor = SGDRegressor
    CartPole_RBF_Network.main()