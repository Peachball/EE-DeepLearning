import theano.tensor as T
import theano
import numpy as np
from DeepLearning import *
import matplotlib.pyplot as plt

class PixelRNN():
    '''
        Follows the paper as closely as possible
        https://arxiv.org/pdf/1601.06759v3.pdf
    '''

    def __init__(self, in_size, out_size, depth, cell_size):
        pass

def load_image(index, dataset='hehexdDataSet'):
    from os.path import join, isfile
    from os import listdir
    files = [f for f in listdir(dataset) if isfile(join(dataset, f))]
    if index >= len(files):
        raise Exception("Invalid index (not enough files)")

    from PIL import Image
    img = Image.open(join(dataset, files[index]))
    img = img.resize((1000, 1000))
    return np.array(img)

def load_images(r, dataset='hehexdDataSet'):
    if not isinstance(r, tuple):
        raise Exception("IDK how to read that non tuple bro")
    if not len(r) == 2:
        raise Exception("incorrect tuple length")

    imgs = []
    for i in range(r[0], r[1]):
        imgs.append(load_image(i, dataset=dataset))

    return np.array(imgs)

def hehexd():
    from theano.tensor.signal.pool import pool_2d
    def display(image):
        print(image.shape)
        plt.imshow(image.transpose(1, 2, 0))
    x = load_images((0, 32))
    x = x.astype(theano.config.floatX) #Prevent bs
    x = x.transpose(0, 3, 1, 2)
    x = x / 255 #Normalize
    X = T.tensor4(name='Images')
    Y = T.matrix(name='output label')

    weights = {}
    layers = {}

    out = None
    def add_conv_layer(inp, shape, subsample, name):
        w = init_weights(shape, init_type='xavier', scale=-1)
        b = init_weights((shape[0],), init_type='uniform', scale=0.001)
        conv = T.nnet.conv2d(inp, w, border_mode='half', subsample=subsample)
        act = T.nnet.relu(conv + b.dimshuffle('x', 0, 'x', 'x'))
        layers[name] = conv
        layers[name + '_act'] = act

        weights[name + '_w'] = w
        weights[name + '_b'] = b
        return act

    def add_pooling(inp, size):
        return pool_2d(inp, size, ignore_border=True)

    def add_layer(inp, shape, name):
        w = init_weights(shape, init_type='xavier')
        b = init_weights(shape[1], init_type='uniform', scale=0.001)
        weights[name + '_w'] = w
        weights[name + '_b'] = b

        res = T.dot(inp, w) + b
        return T.nnet.sigmoid(res)

    output = add_conv_layer(X, (64, 3, 15, 15), (5, 5), 'conv1')
    output = add_conv_layer(output, (128, 64, 3, 3), (1, 1), 'conv2')
    output = add_pooling(output, (2, 2))
    output = add_conv_layer(output, (64, 128, 3, 3), (1, 1), 'conv3')
    output = add_pooling(output, (2, 2))
    output = add_conv_layer(output, (32, 64, 3, 3), (1, 1), 'conv4')
    output = add_pooling(output, (2, 2))
    output = add_conv_layer(output, (16, 32, 3, 3), (1, 1), 'conv5')
    output = add_pooling(output, (2, 2))

    output = T.flatten(output, outdim=2)

    output = add_layer(output, (2304, 1), 'fc6')

    predict = theano.function([X], output)

    error = T.mean(-(Y) * T.log(output) - (1 - Y) * T.log(1 - output))

    params = list(weights.values())
    upd = generateVanillaUpdates(params, error)

    learn = theano.function([X, Y], error, updates=upd,
            allow_input_downcast=True)
    cor = np.ones((32, 1)).astype(theano.config.floatX)
    print(learn(x, cor))

if __name__ == '__main__':
    hehexd()
