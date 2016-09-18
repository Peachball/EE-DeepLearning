import theano.tensor as T
import theano
import numpy as np
from DeepLearning import *
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation, Reshape, BatchNormalization
from keras.layers.convolutional import Convolution2D, UpSampling2D, MaxPooling2D
from keras.optimizers import SGD


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
        index = index % len(files)

    from PIL import Image
    img = Image.open(join(dataset, files[index]))
    img = img.convert('RGB')
    img = img.resize((1024, 1024))
    return np.array(img)

def load_images(r, dataset='hehexdDataSet'):
    if not isinstance(r, tuple):
        raise Exception("IDK how to read that non tuple bro")
    if not len(r) == 2:
        raise Exception("incorrect tuple length")

    imgs = []
    for i in range(r[0], r[1]):
        imgs.append(load_image(i, dataset=dataset))

    return np.array(imgs).astype('float32').transpose(0, 3, 1, 2) / 255

def add_conv_layer(inp, shape, subsample, name, weights, layers, act_type='relu'):
    if name + '_w' in weights:
        w = weights[name + '_w']
    else:
        w = init_weights(shape, init_type='xavier', scale=-1)
    if name + '_b' in weights:
        b = weights[name + '_b']
    else:
        b = init_weights((shape[0],), init_type='uniform', scale=0.001)
    conv = T.nnet.conv2d(inp, w, border_mode='half', subsample=subsample)
    if act_type == 'relu':
        act = T.nnet.relu(conv + b.dimshuffle('x', 0, 'x', 'x'))
    if act_type == 'sigmoid':
        act = T.nnet.sigmoid(conv + b.dimshuffle('x', 0, 'x', 'x'))
    layers[name] = conv
    layers[name + '_act'] = act

    weights[name + '_w'] = w
    weights[name + '_b'] = b
    return act

def add_pooling(inp, size):
    from theano.tensor.signal.pool import pool_2d
    return pool_2d(inp, size, ignore_border=True)

def add_layer(inp, shape, name, weights, layers):
    if name+'_w' in weights:
        w = weights[name + '_w']
    else:
        w = init_weights(shape, init_type='xavier')
    if name+'_b' in weights:
        b = weights[name + '_b']
    else:
        b = init_weights(shape[1], init_type='uniform', scale=0.001)
    weights[name + '_w'] = w
    weights[name + '_b'] = b

    res = T.dot(inp, w) + b
    layers[name] = res
    return T.nnet.sigmoid(res)

def add_upsample(inp, shape):
    o = T.extra_ops.repeat(inp, shape[0], axis=2)
    o = T.extra_ops.repeat(o, shape[1], axis=3)
    return o

def construct_model(inp, weights, layers):
    output = add_conv_layer(inp, (64, 3, 3, 3), (1, 1), 'conv1', weights, layers)
    output = add_pooling(output, (2, 2))
    output = add_conv_layer(output, (128, 64, 3, 3), (1, 1), 'conv2', weights,
            layers)
    output = add_pooling(output, (2, 2))
    output = add_conv_layer(output, (128, 128, 3, 3), (1, 1), 'conv3', weights,
            layers)
    output = add_pooling(output, (2, 2))
    output = add_conv_layer(output, (128, 128, 3, 3), (1, 1), 'conv4', weights,
            layers)
    output = add_pooling(output, (2, 2))
    output = add_conv_layer(output, (64, 128, 3, 3), (1, 1), 'conv5', weights,
            layers)
    output = add_pooling(output, (2, 2))
    output = add_conv_layer(output, (32, 64, 3, 3), (1, 1), 'conv6', weights,
            layers)
    output = add_pooling(output, (2, 2))
    output = add_conv_layer(output, (16, 32, 3, 3), (1, 1), 'conv7', weights,
            layers)
    output = add_pooling(output, (2, 2))
    output = T.flatten(output, outdim=2)

    output = add_layer(output, (1024, 1), 'fc8', weights, layers)

    return output

def construct_generator(inp, weights, layers):
    o = inp
    o = add_layer(o, (100, 4096), 'exp1', weights, layers)
    o = o.reshape((-1, 1024, 2, 2))
    o = add_upsample(o, (2, 2))
    o = add_conv_layer(o, (512, 1024, 3, 3), (1,1), 'deconv2', weights, layers) #4x4
    o = add_upsample(o, (2, 2))
    o = add_conv_layer(o, (256, 512, 3, 3), (1,1), 'deconv3', weights, layers) #8x8
    o = add_upsample(o, (2, 2))
    o = add_conv_layer(o, (128, 256, 3, 3), (1,1), 'deconv4', weights, layers) #16x16
    o = add_upsample(o, (2, 2))
    o = add_conv_layer(o, (128, 128, 3, 3), (1,1), 'deconv5', weights, layers) #32x32
    o = add_upsample(o, (2, 2))
    o = add_conv_layer(o, (128, 128, 3, 3), (1,1), 'deconv6', weights, layers) #64x64
    o = add_upsample(o, (2, 2))
    o = add_conv_layer(o, (128, 128, 3, 3), (1,1), 'deconv7', weights, layers) #128x128
    o = add_upsample(o, (2, 2))
    o = add_conv_layer(o, (64, 128, 3, 3), (1,1), 'deconv8', weights, layers) #256x256
    o = add_upsample(o, (2, 2))
    o = add_conv_layer(o, (32, 64, 3, 3), (1,1), 'deconv9', weights, layers) #512x512
    o = add_upsample(o, (2, 2))
    o = add_conv_layer(o, (3, 32, 3, 3), (1,1), 'deconv10', weights, layers,
            act_type='sigmoid') #1024x1024

    return o

def hehexd():
    def display(image):
        print(image.shape)
        plt.imshow(image.transpose(1, 2, 0))
    x = load_images(tuple([0, 32]))
    X = T.tensor4(name='Images')
    Y = T.matrix(name='output label')
    Z_ = T.matrix(name='seed')

    gen_weights = {}
    gen_layers = {}

    p_weights = {}
    p_layers = {}

    print("Constructing model")
    generated = construct_generator(Z_, gen_weights, gen_layers)
    output = construct_model(T.concatenate([X, generated], axis=0), p_weights,
            p_layers)
    predict_only = construct_model(X, p_weights, p_layers)
    generate = theano.function([Z_], generated, allow_input_downcast=True)

    predict = theano.function([X, Z_], output, allow_input_downcast=True)

    error = T.mean(-(Y) * T.log(output) - (1 - Y) * T.log(1 - output))
    accuracy = T.sum(T.eq(Y, T.round(output)).astype(theano.config.floatX)) / Y.shape[0]
    eval_accuracy = theano.function([X, Z_, Y], accuracy)

    params = list(p_weights.values())
    all_param_map = {}
    all_param_map.update(p_weights)
    all_param_map.update(gen_weights)

    upd = generateVanillaUpdates(params, error, alpha=0.001)

    learn = theano.function([X, Z_, Y], error, updates=upd,
            allow_input_downcast=True)

    savefile = 'bigdcgan.h5'
    errorfile = 'bigdcganerr.txt'
    err = open(errorfile, 'a')
    try:
        loadH5(all_param_map, savefile)
    except Exception as e:
        print("Unable to load previous params")
        print(e)
    # saveH5(all_param_map, savefile)
    image_num = 9
    gen_num = 1
    cor = np.ones((image_num + gen_num, 1)).astype(theano.config.floatX)
    cor[image_num:] = 0
    iteration = 0
    currange = [0, 32]
    while True:
        index = iteration - currange[0]
        error = learn(x[index:index+image_num], np.random.rand(gen_num, 100), cor)
        print(error)
        err.write(str(error) + '\n')
        err.flush()
        iteration += image_num
        if iteration + image_num >= currange[1]:
            currange[0] = currange[1]
            currange[1] = currange[0] + max(50, image_num)
            x = load_images(tuple(currange))
            print("Loading more files")
        if iteration % 100 == 0:
            saveH5(all_param_map, savefile)

def keras_generator():
    pass

if __name__ == '__main__':
    hehexd()
