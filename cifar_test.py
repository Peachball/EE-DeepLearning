import theano.tensor as T
import theano
import tensorflow as tf
from keras.models import Sequential
import numpy as np
from DeepLearning import init_weights
import math


def get_data():
    from keras.datasets import cifar10
    from keras.datasets import cifar100
    (X_train, Y_train), (X_test, Y_test) = cifar100.load_data(label_mode='fine')

    def to_categorical(y):
        z = np.zeros((y.shape[0], y.max()+1))
        indexes = np.arange(z.shape[0])
        z[indexes, y[indexes,0]] = 1
        return z

    Y_train = to_categorical(Y_train)
    Y_test = to_categorical(Y_test)

    #Noramlize pixels to [0, 1]
    X_train /= 255
    X_test /= 255

    return (X_train, Y_train), (X_test, Y_test)

def kerasTest():
    from keras.layers import Convolution2D, MaxPooling2D, Dense, Flatten, Dropout
    from keras.optimizers import SGD
    from keras.models import model_from_yaml
    #Generate model
    model = Sequential()

    model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same',
                        input_shape=(3, 32, 32),))
    model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D((2, 2), border_mode='valid'))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))
    model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D((2, 2), border_mode='valid'))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='softmax'))

    #SGD is known to work (just slow af)
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    try:
        model.load_weights('keras_cifar.h5')
    except Exception as e:
        print("Unable to load previous weights")
        model.save_weights("keras_cifar.h5", overwrite=True)

    #Adam works, but maybe not as well as sgd?
    model.compile(optimizer=sgd, loss='categorical_crossentropy',
            metrics=['accuracy'])

    (x, y), _ = get_data()


    while True:
        model.fit(x, y, nb_epoch=10, validation_split=0.2)

        #Save model and weights
        model.save_weights("keras_cifar.h5", overwrite=True)

def theanoTest():
    theano.config.floatX = 'float32'

    (X_train, Y_train), _ = get_data()

    x_input = T.tensor4() #Dimensions: (examples, channels, rows, columns)
    model = {}
    model["params"] = []
    model['input'] = x_input

    shape = (32, 3, 3, 3)  #(nb_filters, inp_filters, row, columns)
    filt = theano.shared(init_weights(shape), name='conv1_1')
    bias = theano.shared(init_weights(shape[0]), name='conv1_1 bias')
    layer = T.nnet.conv2d(x_input, filt, filter_shape=shape, border_mode='half')
    layer = T.nnet.relu(layer + bias.dimshuffle('x', 0, 'x', 'x'))
    model['params'].append(filt)
    model['params'].append(bias)

    shape = (32, 32, 3, 3)
    filt = theano.shared(init_weights(shape), name='conv1_2')
    bias = theano.shared(init_weights(shape[0]), name='conv1_2 bias')
    layer = T.nnet.conv2d(layer, filt, filter_shape=shape, border_mode='half')
    layer = T.nnet.relu(layer + bias.dimshuffle('x', 0, 'x', 'x'))
    model['params'].append(filt)
    model['params'].append(bias)

    #First Maxpooling
    layer = T.signal.pool.pool_2d(layer, (2, 2), ignore_border=True)

    shape = (64, 32, 3, 3)
    filt = theano.shared(init_weights(shape), name='conv2_1')
    bias = theano.shared(init_weights(shape[0]), name='conv2_1 bias')
    layer = T.nnet.conv2d(layer, filt, filter_shape=shape, border_mode='half')
    layer = T.nnet.relu(layer + bias.dimshuffle('x', 0, 'x', 'x'))
    model['params'].append(filt)
    model['params'].append(bias)

    shape = (64, 64, 3, 3)
    filt = theano.shared(init_weights(shape), name='conv2_2')
    bias = theano.shared(init_weights(shape[0]), name='conv2_2 bias')
    layer = T.nnet.conv2d(layer, filt, filter_shape=shape, border_mode='half')
    layer = T.nnet.relu(layer + bias.dimshuffle('x', 0, 'x', 'x'))
    model['params'].append(filt)
    model['params'].append(bias)

    #Second maxpooling
    layer = T.signal.pool.pool_2d(layer, (2, 2), ignore_border=True)

    layer = T.flatten(layer, 2)

    shape = (4096, 512)
    var = theano.shared(init_weights(shape), name='Dense1')
    bias = theano.shared(init_weights(shape[1]), name='Dense1 bias')
    layer = T.nnet.relu(T.dot(layer, var) + bias)
    model['params'].append(var)
    model['params'].append(bias)

    shape = (512, 10)
    var = theano.shared(init_weights(shape), name='Dense2')
    bias = theano.shared(init_weights(shape[1]), name='Dense2 bias')
    layer = T.nnet.softmax(T.dot(layer, var) + bias)

    model['out'] = layer

    model['predict'] = theano.function([model['input']], [model['out']],
                        allow_input_downcast=True)

    model['target'] = T.matrix()
    model['error'] = T.mean(T.nnet.categorical_crossentropy(model['out'],
                                                    model['target']))


    from DeepLearning import generateAdam
    _, updates = generateAdam(model['params'], model['error'])
    model['learn'] = theano.function([model['input'], model['target']],
                                        model['error'], updates=updates,
                                        allow_input_downcast=True)

    def shuffle():
        indexes = np.random.shuffle(np.arange(X_train.shape[0]))
        X_train = X_train[indexes]
        Y_train = Y_train[indexes]

    def get_accuracy():
        pass

    print("Model is predicting")
    model['predict'](X_train[:10000], Y_train[:10000])
    print("done")

    epochs = 10
    batchsize = 32
    for i in range(epochs):
        for j in range(0, X_train.shape[0], batchsize):
            err = model['learn'](X_train[j:(j+batchsize)], Y_train[j:(j+batchsize)])

            print(err)

def tfTest():
    (X_train, Y_train), _ = get_data()
    x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
    y_ = tf.placeholder(tf.float32, shape=[None, 100])

    def weight_var(shape):
        """Xavier Initalization for a weight var"""
        if len(shape) == 4:#Filter is [height, width, in_channels, out_channels]
            inp = shape[2] * shape[0] * shape[1] + shape[3] * shape[0] *\
                        shape[1]
            scale = math.sqrt(6 / inp)
        elif len(shape) == 2:
            inp = shape[0] + shape[1]
            scale = math.sqrt(2 / inp)
        elif len(shape) == 1:
            inp = shape[0]
            scale = np.sqrt(2 / inp)
        return tf.Variable(tf.random_uniform(shape, minval=-scale,
           maxval=scale))

    conv1_1 = weight_var([3, 3, 3, 32])
    bias1_1 = tf.Variable(tf.random_uniform([32], minval=-0.05,
        maxval=0.05), name="bias1_1")
    layer1_1 = tf.nn.conv2d(x, conv1_1, [1, 1, 1, 1], "SAME") + bias1_1

    conv1_2 = weight_var([3, 3, 32, 32])



    out = layer1_1
    saver = tf.train.Saver()
    init_op = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init_op)
        print(out.eval({x: X_train.transpose(0, 2, 3, 1)[:10]}))

if __name__ == '__main__':
    tfTest()
