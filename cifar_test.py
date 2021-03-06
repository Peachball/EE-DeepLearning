from __future__ import print_function
import theano.tensor as T
import theano
from keras.models import Sequential
import numpy as np
from models.DeepLearning import init_weights
import math
import matplotlib.pyplot as plt
# from DeepLearning import *


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
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    return (X_train, Y_train), (X_test, Y_test)

def kerasTest():
    from keras.layers import Convolution2D, MaxPooling2D, Dense, Flatten, Dropout
    from keras.optimizers import SGD
    from keras.models import model_from_yaml
    #Generate model
    model = Sequential()

    model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same',
                        input_shape=(3, 32, 32)))
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
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=False)

    # try:
        # model.load_weights('keras_cifar.h5')
    # except Exception as e:
        # print("Unable to load previous weights")
        # model.save_weights("keras_cifar.h5", overwrite=True)

    #Adam works, but maybe not as well as sgd?
    model.compile(optimizer=sgd, loss='categorical_crossentropy',
            metrics=['accuracy'])

    (x, y), _ = get_data()

    while True:
        model.fit(x, y, nb_epoch=10, validation_split=0.2)

        #Save model and weights
        # model.save_weights("keras_cifar.h5", overwrite=True)

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
    tf.image_summary("Input images", x)

    def weight_var(shape, name=None):
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
           maxval=scale), name=name)

    conv1_1 = weight_var([3, 3, 3, 32], name="conv1_1")
    bias1_1 = tf.Variable(tf.random_uniform([32], minval=-0.05,
        maxval=0.05), name="bias1_1")
    layer1_1 = tf.nn.relu(tf.nn.conv2d(x, conv1_1, [1, 1, 1, 1], "SAME") +
            bias1_1)

    conv1_2 = weight_var([3, 3, 32, 32], name='conv1_1')
    bias1_2 = tf.Variable(tf.random_uniform([32], minval=-0.05, maxval=0.05),
            name="bias1_1")
    layer1_2 = tf.nn.relu(tf.nn.conv2d(layer1_1, conv1_2, [1, 1, 1, 1], "SAME") +
            bias1_2)

    layer2 = tf.nn.max_pool(layer1_2, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")

    conv3_1 = weight_var([3, 3, 32, 64], name="conv3_1")
    bias3_1 = tf.Variable(tf.random_uniform([64], minval=-0.05, maxval=0.05),
            name="bias3_1")
    layer3_1 = tf.nn.relu(tf.nn.conv2d(layer2, conv3_1, [1, 1, 1, 1], "SAME") +
            bias3_1)

    conv3_2 = weight_var([3, 3, 64, 64], name="conv3_2")
    bias3_2 = tf.Variable(tf.random_uniform([64], minval=-0.05, maxval=0.05),
            name="bias3_2")
    layer3_2 = tf.nn.relu(tf.nn.conv2d(layer3_1, conv3_2, [1, 1, 1, 1], "SAME") +
            bias3_2)

    layer4 = tf.nn.max_pool(layer3_2, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")

    weight5 = weight_var([4096, 512], name="weight5")
    bias5 = tf.Variable(tf.random_uniform([512], minval=-0.05, maxval=0.05),
            name="bias5")
    layer5 = tf.nn.relu(tf.matmul(tf.reshape(layer4, [-1, 4096]), weight5) +
            bias5)

    weight6 = weight_var([512, 100], name="weight6")
    bias6 = tf.Variable(tf.random_uniform([100], minval=-0.05, maxval=0.05),
            name="bias6")
    layer6 = tf.nn.softmax(tf.matmul(layer5, weight6) + bias6)

    out = layer6
    cross_entropy = tf.reduce_mean(tf.reduce_sum(-y_ * tf.log(out),
        reduction_indices=1))

    tf.scalar_summary("Loss over time", cross_entropy)

    correct = (tf.equal(tf.argmax(out, 1), tf.argmax(y_, 1)))
    num_correct = tf.reduce_sum(tf.cast(correct, tf.int32))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    global_step = tf.Variable(0.0)
    alpha = 0.01 / (tf.cast(global_step, tf.float32) * 1e-6 + 1)
    train_op = tf.train.MomentumOptimizer(alpha, 0.9).minimize(cross_entropy)
    saver = tf.train.Saver()
    init_op = tf.initialize_all_variables()
    summary = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter('./tfcifar/cifar_debug/')
    sv = tf.train.Supervisor(init_op=init_op,
                             logdir='./tfcifar/cifar_tf_data',
                             saver=saver,
                             summary_op=summary)

    with sv.managed_session() as sess:
        X_train = X_train.transpose(0, 2, 3, 1)

        while True:
            total = 0
            for i in range(0, X_train.shape[0], 32):
                writer.add_graph(sess.graph)
                _, error, correct, s = sess.run([train_op, cross_entropy,
                    num_correct, summary],
                    feed_dict={x: X_train[i:(i+32)],
                        y_: Y_train[i:(i+32)]})
                writer.add_summary(s)
                total += correct
                print(error, (total * 1.0 / (i+32)))


def get_weight(shape, params=None):
    if len(shape) == 4:
        scale = np.sqrt(6.0) / (shape[1] * shape[2] * shape[3])
    if len(shape) == 2:
        scale = np.sqrt(6.0) / (shape[0] + shape[1])
    if len(shape) == 1:
        scale = 0.05

    var = theano.shared(np.random.uniform(low=-scale, high=scale,
        size=shape).astype(theano.config.floatX))

    if not params is None:
        params.append(var)

    return var

def add_conv_layer(inp, shape, params, bias=True):
    filt = get_weight(shape)
    if bias:
        bias = get_weight((shape[0],))
    conv = T.nnet.conv2d(inp, filt, border_mode='half')

    if bias:
        layer = conv + bias.dimshuffle('x', 0, 'x', 'x')
    else:
        layer = conv

    if bias:
        params += [filt, bias]
    else:
        params += [filt]
    return layer


def hyperconnection_test():
    """
        4 Layer hyperconnected convolution network to be tested against
        cifar100
    """

    theano.config.floatX = 'float32'

    x = T.tensor4()
    y = T.matrix()
    params = []

    b = get_weight((128,))

    layer1 = T.nnet.relu(add_conv_layer(x, (128, 3, 3, 3), params))
    layer1 = T.signal.pool.pool_2d(layer1, (2, 2), ignore_border=True)

    def get_adj_inp(sample_size):
        return T.signal.pool.pool_2d(add_conv_layer(x, (128, 3, 1, 1), params),
                sample_size, ignore_border=True)
    cur_layers = [layer1]

    layer2 = get_adj_inp((2, 2))

    for l in cur_layers:
        layer2 = layer2 + add_conv_layer(l, (128, 128, 3, 3), params,
                bias=False)

    layer2 = T.nnet.relu(layer2)
    layer2 = T.signal.pool.pool_2d(layer2, (2, 2), ignore_border=True)

    cur_layers = [T.signal.pool.pool_2d(l, (2, 2), ignore_border=True) for l in
            cur_layers]

    cur_layers.append(layer2)

    layer3 = get_adj_inp((4, 4))

    for l in cur_layers:
        layer3 = layer3 + add_conv_layer(l, (128, 128, 3, 3), params,
                bias=False)
    layer3 = T.nnet.relu(layer3)
    layer3 = T.signal.pool.pool_2d(layer3, (2, 2), ignore_border=True)

    cur_layers = [T.signal.pool.pool_2d(l, (2, 2), ignore_border=True) for l in
            cur_layers]

    cur_layers.append(layer3)

    layer4 = get_adj_inp((8, 8))
    for l in cur_layers:
        layer4 = layer4 + add_conv_layer(l, (128, 128, 3, 3), params,
                bias=False)

    layer4 = T.nnet.relu(layer4)

    w = get_weight((2048, 100), params)
    b = get_weight((100,), params)
    out = T.nnet.softmax(T.dot(T.flatten(layer4, 2), w) + b)

    predict = theano.function([x], out)

    num_correct = T.sum(T.eq(T.argmax(out, 1), T.argmax(y, 1)))

    cross_entropy = T.mean(T.sum(-y * T.log(T.maximum(out, 1e-20)), 1))

    print("Loading weights")

    (X, Y), _ = get_data()

    (sto, updates) = generateMomentumUpdates(params, cross_entropy, 0.01, 0.9)
    learn = theano.function([x, y], [num_correct, cross_entropy],
            allow_input_downcast=True, updates=updates)

    import sys

    def save():
        saveParams(params, open("hyperconnection_test.npz", "wb"))

    def load():
        loadParams(params, open('hyperconnection_test.npz', 'rb'))

    load()

    epoch = 0
    while True:
        total_cor = 0
        epoch += 1
        print("Epoch:", epoch)
        for i in range(0, X.shape[0], 32):
            [cor, loss] = learn(X[i:(i+32)], Y[i:(i+32)])

            total_cor += cor
            accuracy = total_cor * 1.0 / (i + 32)

            print("\rAccuracy:", accuracy, "Loss:", loss, i, end="")
            sys.stdout.flush()
        print("")
        save()

def keras_control_test():
    from keras.layers import Dense, Convolution2D, MaxPooling2D, Flatten
    from keras.layers import BatchNormalization, Dropout
    from keras.models import Sequential
    from keras.optimizers import SGD

    act = 'softplus'
    model = Sequential()
    model.add(Convolution2D(128, 3, 3, activation=act, border_mode='same',
        input_shape=(3, 32, 32)))
    model.add(BatchNormalization(axis=1))
    model.add(Dropout(0.5))
    model.add(Convolution2D(128, 3, 3, activation=act, border_mode='same'))
    model.add(BatchNormalization(axis=1))
    model.add(Dropout(0.5))
    model.add(Convolution2D(128, 3, 3, activation=act, border_mode='same'))
    model.add(MaxPooling2D((2, 2), border_mode='same'))

    model.add(BatchNormalization(axis=1))
    model.add(Dropout(0.5))
    model.add(Convolution2D(128, 3, 3, activation=act, border_mode='same'))
    model.add(BatchNormalization(axis=1))
    model.add(Dropout(0.5))
    model.add(Convolution2D(128, 3, 3, activation=act, border_mode='same'))
    model.add(BatchNormalization(axis=1))
    model.add(Dropout(0.5))
    model.add(Convolution2D(128, 3, 3, activation=act, border_mode='same'))
    model.add(MaxPooling2D((2, 2), border_mode='same'))

    model.add(BatchNormalization(axis=1))
    model.add(Dropout(0.5))
    model.add(Convolution2D(128, 3, 3, activation=act, border_mode='same'))
    model.add(BatchNormalization(axis=1))
    model.add(Dropout(0.5))
    model.add(Convolution2D(128, 3, 3, activation=act, border_mode='same'))
    model.add(BatchNormalization(axis=1))
    model.add(Dropout(0.5))
    model.add(Convolution2D(128, 3, 3, activation=act, border_mode='same'))
    model.add(MaxPooling2D((2, 2), border_mode='same'))
    model.add(BatchNormalization(axis=1))
    model.add(Dropout(0.5))
    model.add(Convolution2D(128, 3, 3, activation=act, border_mode='same'))

    model.add(Flatten())

    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy',
                metrics=['accuracy'])

    (X_dat, Y_dat), (X_val, Y_val) = get_data()
    # X_dat = X_dat.transpose(0, 3, 1, 2)
    # print(X_dat.shape)

    try:
        model.load_weights('6layertest.h5')
    except Exception as e:
        print("Unable to load previous weights")

    def get_val_accuracy(num=100):
        prediction = model.predict_classes(X_val[:num])
        correct = np.argmax(Y_val[:num], axis=1)
        total_correct = np.sum(prediction == np.argmax(Y_val[:num], axis=1))
        total = num
        return 1.0 * total_correct / total

    while True:
        model.save_weights('6layertest.h5', overwrite=True)
        model.fit(X_dat, Y_dat, nb_epoch=1, validation_split=0.2,
                batch_size=32)

def control_test():
    """
        4 Layer standard conv net with 128 channels and 3x3 filters all the way
        through

        Max pooling between each set of filters
    """

    theano.config.floatX = 'float32'

    x = T.tensor4() #Size: (batches, 3, 32, 32)
    y = T.matrix()

    params = []

    layer1 = T.nnet.relu(add_conv_layer(x, (128, 3, 3, 3), params))
    layer1 = T.signal.pool.pool_2d(layer1, (2, 2), ignore_border=True)

    layer2 = T.nnet.relu(add_conv_layer(layer1, (128, 128, 3, 3), params))
    layer2 = T.signal.pool.pool_2d(layer2, (2, 2), ignore_border=True)

    layer3 = T.nnet.relu(add_conv_layer(layer2, (128, 128, 3, 3), params))
    layer3 = T.signal.pool.pool_2d(layer3, (2, 2), ignore_border=True)

    layer4 = T.nnet.relu(add_conv_layer(layer3, (128, 128, 3, 3), params))
    # layer4 = T.signal.pool.pool_2d(layer4, (2, 2), ignore_border=True)

    layer5 = T.flatten(layer4, 2)

    w = get_weight((2048, 100), params)
    b = get_weight((100,), params)
    layer6 = T.dot(layer5, w) + b

    out = T.nnet.softmax(layer6)
    predict = theano.function([x], [out])

    num_correct = T.sum(T.eq(T.argmax(out, 1), T.argmax(y, 1)))
    cross_entropy = T.mean(T.sum(-y * T.log(out), axis=1))

    alpha = theano.shared(np.array(0.001).astype(theano.config.floatX))
    (storage, updates) = generateAdam(params, cross_entropy, alpha)
    updates.append((alpha, alpha / (1 + 1e-7)))

    learn = theano.function([x, y], [cross_entropy, num_correct],
            updates=updates, allow_input_downcast=True)

    print(params)


    print("Loading data")
    (X_dat, Y_dat), _ = get_data()

    X = X_dat

    print("Training")
    epoch = 1
    import sys
    while True:
        print("Epoch:", epoch)
        epoch += 1
        total_cor = 0
        for i in range(0, X.shape[0], 32):
            [loss, cor] = learn(X[i:(i+32)], Y_dat[i:(i+32)])
            total = i + 32

            total_cor += cor
            print("\rLoss: ", loss, "Accuracy", total_cor * 1.0 / total, i,
                        end="")
            sys.stdout.flush()
        print("")

if __name__ == '__main__':
    keras_control_test()
