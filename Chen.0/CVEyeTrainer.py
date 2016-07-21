from __future__ import print_function
import tensorflow as tf
import sys
import numpy as np

sys.path.append('..')
# from EyeObserver import *

def eyeTrainer():

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

tf.app.flags.DEFINE_string("job_name", "", "Either ps or worker")
tf.app.flags.DEFINE_string("task_index", "", "Index of task (some number)")
FLAGS = tf.app.flags.FLAGS

def load_mnist():
    from keras.datasets import mnist
    (X_train, Y_train), _ = mnist.load_data()
    X_train /= 255

    def to_categorical(y):
        z = np.zeros((y.shape[0], y.max()+1))
        indexes = np.arange(z.shape[0])
        z[indexes, y[indexes]] = 1
        return z

    Y_train = to_categorical(Y_train)

    return (X_train, Y_train)

def distributed_test():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("server", type=int)

    c = tf.constant("HI")

    cluster = tf.train.ClusterSpec({"worker": [
        # "192.168.0.207:4000"
        "localhost:2200",
        "localhost:2201",
        ],
        "ps": [
        "localhost:2202"
        ]
        }
        )

    args = parser.parse_args()
    if args.server == 0:
        server = tf.train.Server(cluster, job_name='worker', task_index=0)
    elif args.server == 1:
        server = tf.train.Server(cluster, job_name='worker', task_index=1)
    elif args.server == 2:
        server = tf.train.Server(cluster, job_name='ps', task_index=0)
        server.join()

    X_data, Y_data = load_mnist()

    X_image = tf.placeholder(tf.float32, [None, 784])
    Y_label = tf.placeholder(tf.float32, [None, 10])

    with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % args.server,
        cluster=cluster)):
        print("Defining device variables n stuff")

        w = tf.Variable(tf.random_uniform([784, 10], minval=-0.05, maxval=0.05))
        b = tf.Variable(tf.random_uniform([10], ))

        y_ = tf.nn.softmax(tf.matmul(X_image, w) + b)

        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),
                                        reduction_indices=[1]))
        train_op = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
        init_op = tf.initialize_all_variables()

        global_step = tf.Variable(0)

    print("Defining supervisor")
    sv = tf.train.Supervisor(is_chief=(FLAGS.task_index==0),
                             init_op=init_op,
                             global_step=global_step)

    print("Done with supervisor")

    with tf.Session(server.target) as sess:
        print("session in session")
        sess.run(init_op)
        for i in range(1000):
            error = sess.run([train_op], feed_dict={X_image: X_data,
                                                    Y_label: Y_data})
            print(error)
            print("Somehting ran!")
        print("ey")

    # with sv.managed_session(server.target):
        # print("Starting managed session")
        # step = 0
        # while not sv.should_stop() and step < 100:
            # print("Training step ran")
            # sess.run([add_op], feed_dict={a: np.random.rand(21, 10),
                                          # b: np.random.rand(21, 10)})
            # step += 1

        # print("Done")
        # sv.stop()

    # with tf.device("/job:worker/task:0"):
        # server0 = tf.constant("Task 0!")

    # with tf.device("/job:worker/task:1"):
        # server1 = tf.constant("Task 1!")

    # with tf.Session("grpc://localhost:2202") as sess:
        # for i in range(10000):
            # sess.run(server0)

def main(_):
    distributed_test()

if __name__ == '__main__':
    tf.app.run()
