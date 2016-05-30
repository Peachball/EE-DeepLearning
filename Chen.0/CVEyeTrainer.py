from __future__ import print_function
import tensorflow as tf
import sys

sys.path.append('..')
from EyeObserver import *

def eyeTrainer():

    data = getData(0)

    #Set up the model
    pic = tf.placeholder(tf.float32, shape=[None, 2])
    target = tf.placeholder(tf.float32, shape=[None, 1])

    w = tf.Variable(tf.random_normal([2, 4]))
    b = tf.Variable(tf.random_normal([4]))

    a1 = tf.sigmoid(tf.matmul(pic, w) + b)

    w2 = tf.Variable(tf.random_normal([4, 1]))
    b2 = tf.Variable(tf.random_normal([1]))

    output = tf.matmul(a1, w2) + b2

    init_op = tf.initialize_all_variables()

    mse = tf.reduce_mean(tf.reduce_sum(tf.square(output - target),
        reduction_indices=1))

    learn = tf.train.AdamOptimizer(0.3).minimize(mse)

    #Set up the server


    with tf.Session() as sess:
        sess.run(init_op)

        x_dat = np.array([[0,0],[1,1],[1,0],[0,1]])
        y_dat = np.array([[0],[0],[1],[1]])

        for i in range(100):
            learn.run(feed_dict={pic: x_dat, target: y_dat})
            print(mse.eval(feed_dict={pic: x_dat, target: y_dat}))


if __name__ == '__main__':
    eyeTrainer()
