import theano.tensor as T
import theano
import numpy as np
from models.DeepLearning import *
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation, Reshape, BatchNormalization, Lambda
from keras.layers import Flatten, Input
from keras.layers.convolutional import Convolution2D, UpSampling2D, MaxPooling2D
from keras.optimizers import SGD
from keras import backend as K


def load_image(index, dataset='hehexdDataSet'):
    from os.path import join, isfile
    from os import listdir
    dataset = 'datasets/' + dataset
    files = [f for f in listdir(dataset) if isfile(join(dataset, f))]
    if index >= len(files):
        index = index % len(files)

    from PIL import Image
    img = Image.open(join(dataset, files[index]))
    img = img.convert('RGB')
    img = img.resize((1024, 1024))
    return np.array(img)

def data_gen(batch_size=32):
    index = 0
    while True:
        im = []
        for i in range(batch_size):
            im.append(load_image(index))
            index += 1
        im = np.array(im).astype('float32') / 255.0
        im = im.transpose(0, 3, 1, 2)
        yield im

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
    if act_type == 'linear':
        act = conv + b.dimshuffle('x', 0, 'x', 'x')
    if act_type == 'softplus':
        act = T.nnet.softplus(conv + b.dimshuffle('x', 0, 'x', 'x'))
    # layers[name] = conv
    # layers[name + '_act'] = act

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
    output = add_conv_layer(inp, (16, 3, 3, 3), (1, 1), 'conv1', weights, layers)
    output = add_pooling(output, (2, 2))
    output = add_conv_layer(output, (64, 16, 3, 3), (1, 1), 'conv2', weights,
            layers)
    output = add_pooling(output, (2, 2))
    output = add_conv_layer(output, (128, 64, 3, 3), (1, 1), 'conv3', weights,
            layers)
    output = add_pooling(output, (2, 2))
    output = add_conv_layer(output, (128, 128, 3, 3), (1, 1), 'conv4', weights,
            layers)
    output = add_pooling(output, (2, 2))
    output = add_conv_layer(output, (128, 128, 3, 3), (1, 1), 'conv5', weights,
            layers)
    output = add_pooling(output, (2, 2))
    output = add_conv_layer(output, (128, 128, 3, 3), (1, 1), 'conv6', weights,
            layers)
    output = add_pooling(output, (2, 2))
    output = add_conv_layer(output, (16, 128, 3, 3), (1, 1), 'conv7', weights,
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
    o = add_conv_layer(o, (16, 64, 3, 3), (1,1), 'deconv9', weights, layers) #512x512
    o = add_upsample(o, (2, 2))
    o = add_conv_layer(o, (3, 16, 3, 3), (1,1), 'deconv10', weights, layers,
            act_type='sigmoid') #1024x1024

    return o

def hehexd():
    def display(image):
        print(image.shape)
        plt.imshow(image.transpose(1, 2, 0))
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
    image_num = 3
    gen_num = 3
    cor = np.ones((image_num + gen_num, 1)).astype(theano.config.floatX)
    cor[image_num:] = 0
    iteration = 0
    plt.ion()
    for d in data_gen(image_num):
        error = learn(d, np.random.rand(gen_num, 100), cor)
        print(error)
        err.write(str(error) + '\n')
        err.flush()
        iteration += 1

        sample_z = np.random.rand(1, 100)
        g = generate(sample_z)[0]
        plt.subplot(121)
        plt.imshow(g.transpose(1, 2, 0))
        plt.subplot(122)
        plt.imshow(d[0].transpose(1, 2, 0))
        plt.pause(0.05)

        if iteration % 100 == 0:
            saveH5(all_param_map, savefile)

def construct_encoder(inp, param, latent_size=8):
    mu = add_conv_layer(
            inp,
            (8, 3, 3, 3),
            (1, 1),
            'conv1',
            param,
            None,
            act_type='linear')

    log_var = add_conv_layer(
            inp,
            (8, 3, 3, 3),
            (1, 1),
            'conv1/var',
            param,
            None,
            act_type='linear')

    return mu, log_var

def construct_decoder(inp, param):
    output = add_conv_layer(inp, (3, 8, 3, 3), (1, 1), 'deconv1', param, None,
            act_type='linear')
    return output

def VAE():
    X = T.tensor4('image')
    Z = T.tensor4('latent sampled')

    enc_param = {}
    dec_param = {}

    z_mean, z_log_var = construct_encoder(X, enc_param)
    rng = T.shared_randomstreams.RandomStreams()
    sampled = rng.normal(z_mean.shape) * T.exp(z_log_var / 2) + z_mean

    reconstructed = construct_decoder(sampled, dec_param)
    custom_reconstruct = construct_decoder(Z, dec_param)

    kl_divergence = - 0.5 * T.mean(
            1 + z_log_var - T.sqr(z_mean) - T.exp(z_log_var))

    error = T.mean(T.sqr(X - reconstructed)) + 0.1 * kl_divergence

    all_param = {}
    all_param.update(enc_param)
    all_param.update(dec_param)

    params = list(all_param.values())
    alpha = theano.shared(np.array(0.1).astype(theano.config.floatX))
    upd = generateVanillaUpdates(params, error, alpha=alpha)

    learn = theano.function([X], error, updates=upd, allow_input_downcast=True)
    generate = theano.function([X], reconstructed, allow_input_downcast=True)
    gen_sample = theano.function([Z],
            custom_reconstruct,
            allow_input_downcast=True)

    plt.ion()
    ema = 10
    prev_err = ema
    i = 0
    for d in data_gen(9):
        e = learn(d)
        ema = (0.99) * ema + 0.01 * e
        print(e)
        g = generate(d[:1])[0]
        plt.imshow(g.transpose(1, 2, 0))
        plt.pause(0.05)
        i += 1
        if i % 20 == 0:
            if ema > prev_err:
                alpha.set_value(alpha.get_value() / 2.0)
                print("Decreasing alpha")
            prev_err = ema

def keras_DCGAN():
    LATENT_DIM = 32
    X = Input(shape=(3, 1024, 1024))
    encoder = Sequential()
    encoder.add(Convolution2D(32, 3, 3, border_mode='same',
        input_shape=(3, None, None)))
    encoder.add(MaxPooling2D((2, 2), border_mode = 'same')) # 512
    encoder.add(Convolution2D(64, 3, 3, border_mode='same'))
    encoder.add(MaxPooling2D((2, 2), border_mode = 'same')) # 256
    encoder.add(Convolution2D(96, 3, 3, border_mode='same'))
    encoder.add(MaxPooling2D((2, 2), border_mode = 'same')) # 128
    encoder.add(Convolution2D(128, 3, 3, border_mode='same'))
    encoder.add(MaxPooling2D((2, 2), border_mode = 'same')) # 64
    encoder.add(Convolution2D(128, 3, 3, border_mode='same'))
    encoder.add(MaxPooling2D((2, 2), border_mode = 'same')) # 32

    def sample(m):
        m, log_var = m
        norm = K.random_normal(shape=(32, LATENT_DIM, 32, 32),
                mean=0.,
                std=1.)
        return norm * K.exp(log_var / 2) + m

    z_log_var = Convolution2D(LATENT_DIM, 3, 3, border_mode='same')(encoder(X))
    z_mean = Convolution2D(LATENT_DIM, 3, 3, border_mode='same')(encoder(X))
    z = Lambda(
            sample,
            output_shape=(LATENT_DIM, 32, 32))([z_mean, z_log_var])

    decoder = Sequential()
    decoder.add(Convolution2D(LATENT_DIM, 3, 3, border_mode='same',
        input_shape=(3, None, None)))
    decoder.add(UpSampling2D((2, 2)))
    decoder.add(Convolution2D(128, 3, 3, border_mode='same'))
    decoder.add(UpSampling2D((2, 2)))
    decoder.add(Convolution2D(128, 3, 3, border_mode='same'))
    decoder.add(UpSampling2D((2, 2)))
    decoder.add(Convolution2D(96, 3, 3, border_mode='same'))
    decoder.add(UpSampling2D((2, 2)))
    decoder.add(Convolution2D(64, 3, 3, border_mode='same'))
    decoder.add(UpSampling2D((2, 2)))
    decoder.add(Convolution2D(32, 3, 3, border_mode='same'))
    decoder.add(UpSampling2D((2, 2)))
    decoder.add(Convolution2D(3, 3, 3, border_mode='same'))

    encoder.compile(optimizer='sgd', loss='binary_crossentropy')

    for d in data_gen():
        pass

if __name__ == '__main__':
    VAE()
