from __future__ import print_function
import sys
import pickle
from collections import OrderedDict
import theano.tensor as T
import theano
from theano.compile.nanguardmode import NanGuardMode
from theano import config
import numpy as np
import struct
import matplotlib.pyplot as plt
import time


def readMNISTData(length=10000):
    images = open('train-images-idx3-ubyte', 'rb')
    labels = open('train-labels-idx1-ubyte', 'rb')
    images.read(8)
    labels.read(8)
    def readInt(isn=True):
        if isn:
            return struct.unpack('>i', images.read(4))[0]
        else:
            return int.from_bytes(labels.read(4), byteorder='big', signed=True)
    xsize = readInt()
    ysize = readInt()
    def readImage():
        img = []
        for i in range(xsize):
            for j in range(ysize):
                img.append(struct.unpack('>B', images.read(1))[0])
        return img

    def readLabel():
        testLabel = [0]*10
        testLabel[struct.unpack('B', labels.read(1))[0]] = 1
        return testLabel

    imgs = []
    lbls = []
    for i in range(length):
        imgs.append(readImage())
        lbls.append(readLabel())
        print('\rRead {}/{}'.format(i+1, length), end="")
    print('\nDone reading')
    return (np.array(imgs), np.array(lbls))

def readcv(length=10000):
    images = open('t10k-images-idx3-ubyte', 'rb')
    labels = open('t10k-labels-idx1-ubyte', 'rb')
    images.read(8)
    labels.read(8)
    def readInt(isn=True):
        if isn:
            return struct.unpack('>i', images.read(4))[0]
        else:
            return int.from_bytes(labels.read(4), byteorder='big', signed=True)
    xsize = readInt()
    ysize = readInt()
    def readImage():
        img = []
        for i in range(xsize):
            for j in range(ysize):
                img.append(struct.unpack('>B', images.read(1))[0])
        return img

    def readLabel():
        testLabel = [0]*10
        testLabel[struct.unpack('B', labels.read(1))[0]] = 1
        return testLabel

    imgs = []
    lbls = []
    for i in range(length):
        imgs.append(readImage())
        lbls.append(readLabel())
        print('\rRead {}/{}'.format(i+1, length), end="")

    print ('\nDone Reading')
    return (np.array(imgs), np.array(lbls))


def reset(params, init_size=0.1, init_range=None):
    if init_range == None:
        init_range = (-init_size, init_size)
    for p in params:
        p.set_value(np.random.uniform(low=init_range[0], high=init_range[1],
            size=p.shape.eval()).astype(theano.config.floatX))

def init_weights(shape, init_type='uniform', scale=-1, shared_var=True,
        name=None):
    if init_type == 'uniform':
        DEFAULT_SCALE = 0.05
        if scale < 0:
            scale = DEFAULT_SCALE
        val = np.random.uniform(low=-scale, high=scale, size=shape).astype(
                theano.config.floatX)
        if shared_var:
            return theano.shared(val, name=name)
        else:
            return val

    if init_type == 'bias':
        DEFAULT_SCALE = 1

        if scale < 0:
            scale = DEFAULT_SCALE

        var = np.random.uniform(low=scale-0.001, high=scale+0.001,
                size=shape).astype(theano.config.floatX)
        if shared_var:
            return theano.shared(var, name=name)
        else:
            return var

    if init_type == 'xavier':
        DEFAULT_SCALE = 6
        in_neurons = 0
        out_neurons = 0

        if scale < 0:
            scale = DEFAULT_SCALE

        if isinstance(shape, int) or len(shape)==1:
            # Shape 1 means that it is bias, therefore the initialization
            # doesn't really matter
            in_neurons = 400 #To get s approx. eq. to 0.05
            out_neurons = 0
            scale = 1

        elif len(shape) == 2:
            in_neurons = shape[0]
            out_neurons = shape[1]

        #Convolution: [output_channels, input_channels, rows, columns]
        elif len(shape) == 4:
            in_neurons = shape[1] * shape[2] * shape[3]
            out_neurons = shape[0] * shape[2] * shape[3]

        s = np.sqrt(scale * 1.0 / (in_neurons + out_neurons))

        if not shared_var:
            return np.random.uniform(low=-s, high=s, size=shape).astype(
                    theano.config.floatX)
        else:
            return theano.shared(np.random.uniform(low=-s, high=s,
                size=shape).astype(theano.config.floatX), name=name)

    if init_type == 'zeros':
        if not shared_var:
            return np.zeros(shape).astype(theano.config.floatX)
        else:
            return theano.shared(np.zeros(shape).astype(theano.config.floatX))


def generateHessian(params, error, alpha=1, verbose=False):
    if verbose:
        print("Calculating Updates")

    updates = []

    for p in params:
        grad = T.grad(error, p)
        hessian = theano.gradient.hessian(error, p)
        updates.append((p, p - alpha * T.dot(T.inv(hessian), grad)))
    return updates

def generateAdagrad(params, error, alpha=0.01, epsilon=1e-8, verbose=False,
        clip=None):
    updates = []
    history = []

    if verbose:
        print("Calculating gradients")
    gradients = T.grad(error, params)
    if verbose:
        print("Done with gradients")
    count = 0
    for p, grad in zip(params, gradients):
        shape = p.get_value().shape

        totalG = theano.shared(value=np.zeros(shape).astype(theano.config.floatX))

        new_g = totalG + T.sqr(grad)
        updates.append((totalG, new_g))
        deltaw = grad / (T.sqrt(new_g) + epsilon) * alpha
        if isinstance(clip, tuple):
            deltaw = T.clip(deltaw, clip[0], clip[1])
        updates.append((p, p - deltaw))

        history.append(totalG)

    if verbose: print('')
    return (history, updates)

def generateAdadelta(params, error, decay=0.9, alpha=1, epsilon=1e-8):
    updates = []
    accUpdates = []
    accGrad = []

    gradients = T.grad(error, params)
    for p, grad in zip(params, gradients):
        shape = p.get_value().shape

        Eg = theano.shared(value=np.zeros(shape).astype(theano.config.floatX))
        Ex = theano.shared(value=np.zeros(shape).astype(theano.config.floatX))

        new_g = decay * Eg + (1 - decay) * T.sqr(grad)

        d_x = T.sqrt((Ex + epsilon) / (new_g + epsilon)) * grad * alpha
        new_x = decay * Ex + (1 - decay) * T.sqr(d_x)

        updates.append((p, p - d_x))
        updates.append((Ex, new_x))
        updates.append((Eg, new_g))

        accUpdates.append(Ex)
        accGrad.append(Eg)

    return ([accUpdates, accGrad], updates)

def generateAdam(params, error, alpha=0.001, decay1=0.9, decay2=0.999,
        epsilon=1e-8, verbose=False):
    """
        Generate updates for the adam type of stochastic gradient descent

            Variable interp.
    """
    updates = []
    moment = []
    vector = []

    if type(alpha) == 'float':
        alpha = theano.shared(np.array(alpha).astype(theano.config.floatX))
    time = theano.shared(np.array(1.0).astype(theano.config.floatX))
    epsilon = theano.shared(np.array(epsilon).astype(theano.config.floatX))
    updates.append((time, time+1))
    i = 0
    gradients = T.grad(error, params)
    decay1 = theano.shared(np.array(decay1).astype(theano.config.floatX))
    decay2 = theano.shared(np.array(decay2).astype(theano.config.floatX))
    for p, grad in zip(params, gradients):
        shape = p.get_value().shape
        grad = T.grad(error, p)

        m = theano.shared(value=np.zeros(shape).astype(theano.config.floatX))
        v = theano.shared(value=np.zeros(shape).astype(theano.config.floatX))

        m_t = decay1 * m + (1 - decay1) * grad
        v_t = decay2 * v + (1 - decay2) * T.sqr(grad)
        m_adj = m_t / (1.0 - T.pow(decay1, time))
        v_adj = v_t / (1.0 - T.pow(decay2, time))

        updates.append((m, m_t))
        updates.append((v, v_t))
        # updates.append((p, p - alpha * m_adj / (T.sqrt(v_adj) + epsilon)))

        moment.append(m)
        vector.append(v)
        if verbose: print("\rDone with {}/{}".format(i+1, len(params)), end="")
        i += 1

    if verbose: print("")
    return (moment + vector + [time, alpha, epsilon], updates)

def generateRmsProp(
        params, error, alpha=0.01, decay=0.9, fudge=1e-3, verbose=False):
    r = []
    v = []
    pr = []
    updates = []
    alpha = theano.shared(np.array(alpha).astype(theano.config.floatX))
    count = 0
    for p in params:
        grad = T.grad(error, p)

        shape = p.get_value().shape
        r_t = theano.shared(np.zeros(shape).astype(theano.config.floatX))
        v_t = theano.shared(np.zeros(shape).astype(theano.config.floatX))

        new_r = (1 - decay) * T.sqr(grad) + decay * r_t
        new_v = alpha / (T.sqrt(new_r) + fudge) * grad
        updates.append((r_t, new_r))
        updates.append((v_t, new_v))
        updates.append((p, p - new_v))
        r.append(r_t)
        v.append(v_t)

        count += 1
        if verbose: print("\rGradient {}/{} done".format(count, len(params)),
                end="")

    if verbose: print('')
    return (r + v, updates)

def generateVanillaUpdates(params, error, alpha=0.01, verbose=True):
    grad = []
    count = 0
    for p in params:
        grad.append(T.grad(error, p))
        count += 1
        print("\r{}/{} Gradients done".format(count, len(params)), end="")
    updates = [(p, p - g * alpha) for p, g in zip(params, grad)]
    print("")
    return updates

def generateMomentumUpdates(params, error, alpha=0.01, momentum=0.5):
    grad = []
    if type(alpha) == float:
        alpha = theano.shared(np.array(alpha).astype(theano.config.floatX))
    if type(momentum) == float:
        momentum = theano.shared(np.array(momentum)
                                    .astype(theano.config.floatX))
    for p in params:
        grad.append(T.grad(error, p))
    mparams = [theano.shared(np.zeros(p.shape.eval()).astype(theano.config.floatX)) for p in params]
    gradUpdates = [(p, p - g) for p, g in zip(params,mparams)]

    gradUpdates += [(m, momentum * m + alpha * g) for m, g in
        zip(mparams, grad)]
    return ([gradUpdates, mparams], gradUpdates)

def generateNesterovMomentumUpdates(params, error, alpha=0.01, momentum=0.9,
        decay=1e-6):
    print("WARNING: NOT FULLY IMPLEMENTED YET")
    updates = []
    m = []
    for p in params:
        v_t = theano.shared(np.zeros(p.get_value().shape)
                .astype(theano.config.floatX))
        grad = T.grad(error, p + momentum * v_t)

        updates.append((p, p + v_t))
        updates.append((v_t, momentum * v_t - alpha * grad))

        m.append(v_t)

    return (m, updates)

def generateRpropUpdates(params, error, init_size=1, verbose=False):
    prevw = []
    deltaw = []
    updates = []
    gradients = []
    #initalize stuff
    for p in params:
        prevw.append(theano.shared(np.zeros(p.shape.eval()).astype(config.floatX)))
        deltaw.append(theano.shared(init_size * np.ones(p.shape.eval()).
            astype(config.floatX)))

    iterations = 0
    for p, dw, pw in zip(params, deltaw, prevw):
        try:
            if verbose: print("\rGradient {} out of {}".format(iterations + 1, len(params)), end="")
            gradients.append(T.grad(error, p))
            iterations += 1
        except Exception:
            print('Unused input')
            continue
        #Array describing which values are when gradients are both positive or both negative
        simW = T.neq((T.eq((pw > 0), (gradients[-1] > 0))), (T.eq((pw < 0), (gradients[-1] <
            0))))

        #Array describing which values are when gradients are in opposite directions
        diffW = ((pw > 0) ^ (gradients[-1] > 0)) * (T.neq(pw, 0) * T.neq(gradients[-1], 0))
        updates.append((p, p - (T.sgn(gradients[-1]) * dw * (T.eq(diffW, 0)))))
        updates.append((dw, T.switch(diffW, dw *
            0.5, T.switch(simW, dw * 1.2, dw))))
        updates.append((pw, (T.sgn(gradients[-1]) * dw * (T.eq(diffW, 0)))))

    storage = prevw + deltaw
    if verbose: print("\nDone with updates")

    return (storage, updates)

def getRegularization(params):
    reg = T.sum(T.sqr(params[0]))
    for p in params[1:]:
        reg = reg + T.sum(T.sqr(p))
    return reg


class Layer:
    def __init__(self, in_size, out_size, layer_type='sigmoid', in_var=None,
            init_size=0.1, nonlinearity=None, weights=None, init_range=None):
        self.in_size = in_size
        self.out_size = out_size
        self.nonlinearity = nonlinearity
        if init_range == None:
            init_range = (-init_size, init_size)
        if (
                not layer_type in ['sigmoid', 'tanh', 'lstm', 'rnn', 'linear',
                    'rlu', 'rigid_rlu'] 
                and nonlinearity==None):
            raise Exception('Layer type is invalid: ' + str(layer_type))

        if nonlinearity==None:
            if layer_type == 'sigmoid':
                nonlinearity = T.nnet.sigmoid
            if layer_type == 'tanh':
                nonlinearity = T.tanh
            if layer_type == 'linear':
                nonlinearity = lambda x: x
            if layer_type == 'rlu':
                nonlinearity = lambda x: T.log(1 + T.exp(x))
            if layer_type =='rigid_rlu':
                nonlinearity = lambda x: T.clip(x, 0, np.inf)

        if in_var==None:
            x = T.matrix('Input')
        else:
            x = in_var

        if not weights:
            self.w = theano.shared(
                    np.random.uniform(low=init_range[0], high=init_range[1],
                    size=(in_size, out_size)).astype(theano.config.floatX))
        else:
            self.w = weights
        self.b = theano.shared(value=np.random.uniform(low=init_range[0], high=init_range[1],
            size=(out_size)).astype(theano.config.floatX))
        self.out = nonlinearity(T.dot(x, self.w) + self.b)
        self.params = [self.w, self.b]

    def getOutput(self, x,  nonlinearity=T.nnet.sigmoid):
        out = nonlinearity(T.dot(x, self.w) + self.b)
        return out


class AutoEncoder:
    def __init__(self, *dim, **kwargs):
        init_size = kwargs.get('init_size', 1)
        verbose = kwargs.get('verbose', False)
        in_type = kwargs.get('in_type', 'linear')
        layers = []

        self.x = kwargs.get('in_var', T.matrix('Generalized Input'))
        layers.append(Layer(dim[0], dim[1], in_var=self.x, init_size=init_size))
        for i in range(1, len(dim) - 1):
            layers.append(Layer(dim[i], dim[i+1], 
                in_var=layers[-1].out,
                init_size=init_size))

        self.out = layers[-1].out
        decoder = []

        if len(dim) > 2:
            layer_type = 'sigmoid'
        else:
            layer_type = in_type
        decoder.append(Layer(dim[-1], dim[-2],
            in_var = self.out,
            init_size = init_size,
            layer_type = layer_type))
        for i in range(2, len(dim)):
            if i == len(dim) - 1:
                layer_type = in_type
            decoder.append(Layer(dim[-i], dim[-i-1], in_var=decoder[-1].out,
                init_size=init_size, layer_type=layer_type))


        self.reconstructed = decoder[-1].out

        self.encode = layers
        self.decode = decoder

        if verbose: print(len(layers), len(decoder))
        self.layers = self.encode + self.decode

        self.params =[]

        for l in self.layers:
            self.params = self.params + l.params

        if verbose: print("Finished Initialization")

    def learn(self,x, y, mode='vanilla', iterations=10):
        def getEncDecPair(layer):
            encoder = self.encode[layer]
            decoder = self.decode[-(layer + 1)]
            return (encoder, decoder)

        def getLearningFunction(layer, x, y):
            encoder, decoder = getEndDecPair(layer)

            newOut = encoder.getOutput(x)
            out = decoder.getOutput(newOut)

            mse = T.mean(T.sqr(out - y))

            return (x, mse)

        for i in range(self.encode):
            for j in range(iterations):
                pass

        #Warning: Not done at all
class ConvolutionLayer:
    def __init__(self, shape, in_var=T.tensor4('input'),
            nonlinearity=T.nnet.sigmoid, init_size=0.1,
            deconv=False, subsample=None, init_range=None):
        if nonlinearity == None:
            nonlinearity = lambda x: x
        if subsample == None:
            subsample = (1, 1)
        if init_range==None:
            init_range=(-init_size, init_size)
        self.shape = shape
        x = in_var
        self.x = x
        filt = theano.shared(value=(
            np.random.uniform(low=init_range[0], high=init_range[1],
                size=shape)).astype(theano.config.floatX))

        #If bias needs to be applied to every hidden unit, it should be 3d
        bias = theano.shared(value=np.random.uniform(
            low=init_range[0], high=init_range[1], size=shape[0])
            .astype(theano.config.floatX))
        if deconv:
            subsample = (1,1)
            z = T.nnet.conv2d(x, filt, border_mode='full', subsample=subsample)
        else:
            z = T.nnet.conv2d(x, filt, border_mode='valid', subsample=subsample)

        self.out = nonlinearity(z + bias.dimshuffle('x', 0, 'x', 'x'))
        self.w = filt
        self.b = bias
        self.params = [filt, bias]

class ConvolutionalAutoEncoder:
    '''
    Reminder: The goal of this is purely to make dreams, nothing else
    '''
    def __init__(self, *dim, **kwargs):

        x = T.tensor4('input')
        self.x = x
        init_size = kwargs.get('init_size', 0.1)
        out_nonlinearity = kwargs.get('out_nonlinearity', None)
        int_nonlinearity = kwargs.get('int_nonlinearity', T.nnet.sigmoid)
        init_range = kwargs.get('init_range', None)
        if init_range==None:
            init_range = (-init_size, init_size)

        encode = []
        encode.append(ConvolutionLayer(dim[0], in_var=x, init_size=init_size, deconv=False))
        for i in range(1, len(dim)):
            encode.append(ConvolutionLayer(dim[i], in_var=encode[-1].out,
                init_size=init_size, deconv=False,
                nonlinearity=int_nonlinearity))

        decoder = []
        self.out = encode[-1].out

        def swapElements(dimension):
            d = list(dimension)
            d[0], d[1] = d[1], d[0]
            return tuple(d)

        if len(encode) == 1:

            decoder.append(ConvolutionLayer(swapElements(dim[-1]), in_var=encode[-1].out, init_size=init_size,
                deconv=True, nonlinearity=out_nonlinearity))
        else:
            decoder.append(ConvolutionLayer(swapElements(dim[-1]), in_var=encode[-1].out, init_size=init_size,
                deconv=True, nonlinearity=int_nonlinearity))
        #gotta implement the decoding step lol
        for i in range(2, len(dim)):
            nonlinearity = int_nonlinearity
            if i == len(dim) - 1:
                nonlinearity = None
            decoder.append(ConvolutionLayer(swapElements(dim[-i]), in_var=decoder[-1].out, init_size=init_size,
                deconv=True, nonlinearity=out_nonlinearity))

        self.reconstructed = decoder[-1].out

        layers = encode + decoder
        self.layers = layers

        self.encode = encode
        self.decode = decoder

        self.params = []
        for l in layers:
            self.params += l.params

class BNLayer:
    """
        Batch Normalization layer, as descrbed by the original paper
    """
    def __init__(self, shape, in_var=T.matrix(), axis=0):
        """
            Shape contains all dimensions excluding first one
        """
        if isinstance(shape, tuple):
            raise Exception("Invalid shape for BN Layer")
        self.x = in_var
        x = in_var
        mean = T.mean(x, axis=axis)
        variance = T.sqr(x - mean)

        self.mean = mean
        self.variance = variance

        self.beta = init_weights(shape, init_type='uniform')
        self.gamma = init_weights(shape, init_type='uniform')

        self.params = [self.beta, self.gamma]

    def getTrainingOutput(self):
        x_ = (self.x - mean) / T.sqrt(variance)
        return x_ * self.gamma + self.beta

    def getInferenceOutput(self, means, variance):
        x_ = (self.x - means) / T.sqrt(variance)
        return x_ * self.gamma + self.beta


class FFClassifier:
    def __init__(self, *dim, **kwargs):

        x = T.matrix('input')
        self.x = x
        init_size = kwargs.get('init_size', 0.1)

        layers = []
        layers.append(Layer(dim[0], dim[1], in_var=x, init_size=init_size))
        for i in range(1, len(dim) - 1):
            layers.append(Layer(dim[i], dim[i+1], in_var=layers[-1].out, init_size=init_size))

        self.out = layers[-1].out
        self.layers = layers

        self.params = []
        for l in layers:
            self.params = self.params + l.params

def miniBatchLearning(x, y, batchSize, updateFunction, verbose=False, epochs=1):
    train_error = []
    np.set_printoptions(precision=10)
    if batchSize <= 0:
        batchSize = x.shape[0]
    for j in range(epochs):
        for i in range(0, x.shape[0], batchSize):
            error = updateFunction(x[i:(i+batchSize)], y[i:(i+batchSize)])
            train_error.append(error)
            if verbose: print('{0:10} Epoch: {1:4}'.format(np.round(error, 10), round((j +
                (i/x.shape[0])), 2)))
    return train_error

def saveParams(params, f):
    """
        params is a list of theano shared variables
        f is a file name or actual file
    """
    print("Saving...Do not close")
    arr = []
    for p in params:
        arr.append(p.get_value())
    np.savez(f, *arr)
    print("Done!")

def saveH5(param_map, filename):
    import h5py
    print("Saving params")
    f = h5py.File(filename, 'w')
    for k, v in param_map.items():
        f.create_dataset(k, data=v.get_value())

    f.close()
    print("Done")
    return

def loadH5(param_map, filename):
    import h5py
    print("Loading params")
    f = h5py.File(filename, 'r')
    for k, v in param_map.items():
        param_map[k].set_value(f[k][:])
    print("Done")
    return

def loadParams(params, f):
    def load_npz(npz):
        j = {}
        for i in npz:
            j[int(i.replace('arr_', ''))] = npz[i]
        return j
    p = load_npz(np.load(f))
    if len(params) != len(p):
        raise "Paramater length mismatch"
    for par, n in zip(params, p):
        par.set_value(p[n])

def smartTrainer(x, y, batchSize, maxepochs=10, verbose=False, patience=2,
        decrease_rate=0.95):
    pass

def KerasAETester():
    from keras.models import Model
    from keras.layers import Input, Dense
    from keras.optimizers import RMSprop

    images, labels = readMNISTData(10000)
    xcv, ycv = readcv(100)

    picture = Input(shape=(784,))

    encoder = Dense(500, activation='relu')(picture)

    decoder = Dense(784, activation='linear')(encoder)

    model = Model(input=picture, output=decoder)

    model.compile(RMSprop(lr=0.01), loss='mse')

    model.fit(images, images, verbose=1)
    for i in range(len(images)):
        generated_image = model.predict(images[i].reshape(1, 784))
        plt.imshow(generated_image.reshape(28, 28), cmap='Greys',
                interpolation='none')
        plt.figure()
        plt.imshow(images[i].reshape(28, 28), cmap='Greys', interpolation='none')
        plt.show()

def AETester():
    images, labels = readMNISTData(1000)
    xcv, ycv = readcv(100)
    ae = AutoEncoder(784, 500, 300, init_size=0.1, in_type='sigmoid')


    images = images / images.max()

    genImage = theano.function([ae.x], ae.reconstructed,
            allow_input_downcast=True)

    y = T.matrix('correct output')
    yl = T.matrix('Correct labels')
    regError = getRegularization(ae.params)
    mse = T.mean(T.sqr(y - ae.reconstructed))
    normErr = mse

    crossEntrop = -T.mean(y * T.log(ae.reconstructed) +
            (1 - y) * T.log(1 - ae.reconstructed))

    (adamStorage, adam) = generateAdam(ae.params, mse, alpha=0.001)
    (adaStorage, adaGrad) = generateAdagrad(ae.params, mse, alpha=0.01)
    (rstorage, rprop) = generateRpropUpdates(ae.params, mse,
            init_size=0.01)
    (rmsstorage, rms) = generateRmsProp(ae.params, mse, alpha=0.01)

    learn = theano.function([ae.x, y], mse, updates=rms,
            allow_input_downcast=True)
    train_error = miniBatchLearning(images, images, 100, learn, verbose=True,
            epochs=100)

    plt.plot(np.arange(len(train_error)), train_error)
    plt.show()

    for i in range(len(images)):
        generated_image =genImage(images[i].reshape(1, 784))
        print("Min: {0:10} Max: {1:10}".format(generated_image.min(), generated_image.max()))
        plt.imshow(generated_image.reshape(28, 28), cmap='Greys', interpolation='none')
        plt.figure()
        plt.imshow(images[i].reshape(28, 28), cmap='Greys', interpolation='none')
        plt.show()

def simple_AETester():
    ae = AutoEncoder(2, 2, in_type='linear', init_size=0.1)

    data = np.array([[1, 0], [0, 1], [0, 0], [1, 1]])

    predict = theano.function([ae.x], ae.reconstructed)

    y = T.matrix('target')

    error = T.mean(T.sqr(y - ae.reconstructed))

    sgd = generateVanillaUpdates(ae.params, error, alpha=0.01)
    (storage, rprop) = generateRpropUpdates(ae.params, error, init_size=1)

    learn = theano.function([ae.x, y], error, updates=rprop)

    miniBatchLearning(data, data, -1, learn, verbose=True, epochs=1000)

    output = predict(data)
    plt.scatter(data[:,0], data[:,1], color='green')
    plt.scatter(output[:,0], output[:,1], color='red')
    plt.show()

def NNTester():
    images, labels = readMNISTData(6000)
    xcv, ycv = readcv(1000)

    y = T.matrix('Correct Labels')
    nn = FFClassifier(784, 1000, 700, 500, 300, 10)
    error = -T.mean(y * T.log(nn.out) + (1-y) * T.log(1- nn.out))

    dupdates = generateVanillaUpdates(nn.params, error, alpha=0.01)
    (s, adam) = generateAdam(nn.params, error, alpha=0.001)
    (st, rprop) = generateRpropUpdates(nn.params, error, 0.1)
    (sto, rms) = generateRmsProp(nn.params, error, alpha=0.001, decay=0.9)
    (stor, adadelta) = generateAdadelta(nn.params, error, alpha=1, decay=0.9)
    (stora, adagrad) = generateAdagrad(nn.params, error, alpha=0.01)

    updateRules = [(dupdates, "SGD"), (adam, "Adam"), (rms, "RMS"), (adadelta, "Adadelta"), (adagrad, "Adagrad")]

    figure, ax = plt.subplots()
    for u in updateRules:
        reset(nn.params)
        learn = theano.function([nn.x, y], error, updates=u[0], allow_input_downcast=True)
        start_time = time.perf_counter()
        train_error = miniBatchLearning(images, labels, 500, learn, verbose=False, epochs=50)
        print(u[1], " took ", (time.perf_counter() - start_time))
        ax.plot(np.arange(len(train_error)), train_error, label=u[1])

    plt.yscale('log', nonposy='clip')
    ax.legend(loc='upper right')
    plt.xlabel("Iterations")
    plt.ylabel("MSE")
    figure.suptitle("Learning Style Comparison")
    plt.savefig("test.png")
    plt.show()

def HessianTester():
    images, labels = readMNISTData(6000)
    model = FFClassifier(784, 300, 10)
    y = T.matrix('labels')
    error = - T.mean(y * T.log(model.out) + (1-y) * T.log(1 - model.out))
    updates = generateHessian(model.params, error, alpha=0.01)

    learn = theano.function([model.x, y], error, updates=updates,
            allow_input_downcast=True)
    err = miniBatchLearning(images, labels, 1, learn, verbose=True, epochs=5)
    plt.plot(err)
    plt.show()

def normalize(x, dim=0, low=-1, high=1, scaleFactor=None, type='range'):
    if low >= high:
        raise Exception("Bad boundaries")
    if type=='range':
        if scaleFactor == None:
            mins = x.min(axis=dim)
            maxes = x.max(axis=dim)
            eq = np.equal(mins, maxes)
            r = np.where(eq, np.ones(maxes.shape), maxes - mins)
        if isinstance(scaleFactor, tuple):
            maxes, mins = scaleFactor
        return ((maxes, mins),
                (((x - mins) / (r) * (high - low)) + low))

    if type=='gauss':
        if scaleFactor == None:
            means = x.mean(axis=dim)
            var = np.maximum(np.square(x - means).mean(axis=dim), 1e-8)
        if isinstance(scaleFactor, tuple):
            means, var = scaleFactor
        return ((means, var), ((x - means) / np.sqrt(var)))

def scaleBack(x, scale, dim=0, type='range'):
    if type == 'range':
        high, low = scale
        mins = np.min(x, axis=dim)
        maxes = np.max(x, axis=dim)
        eq = np.equal(mins, maxes)
        range = np.where(eq, np.ones(maxes.shape), maxes - mins)
        return (((x - mins) / (range) * (high - low)) + low)
    if type == 'gauss':
        means, var = scale
        return (x * np.sqrt(var) + means)

from PIL import Image

def convertImageToArray(index, size=(100, 100)):
    from os.path import join, isfile
    from os import listdir
    files = [join('hehexdDataSet', f) for f in listdir('hehexdDataSet') if isfile("hehexdDataSet/"+f)]
    im = Image.open(files[index])
    if not size is None:
        im = im.resize(size)
    im.load()
    return np.asarray(im)

def ConvolutionDreamerTest():
    conv = ConvolutionalAutoEncoder((3, 3, 15, 15), init_size=0.01,
            out_nonlinearity=lambda x: T.maximum(0.01 * x, x),
            int_nonlinearity=lambda x: T.maximum(0.01*x, x),
            init_range=(0.5, 0.6))

    i = 0

    def googleImageDownloader(start=0):
        from apiclient.discovery import build
        #nonlocal i

        service = build("customsearch", "v1",
               developerKey="AIzaSyBkd2lwAWzEKWjRbB2rlRM5OU_OBvz7u5w")

        res = service.cse().list(
            q='flower',
            cx='015897475084883013797:9dpu7012l6o',
            searchType='image',
            num=10,
            start=start,
            imgType='photo',
            fileType='png',
            imgSize='huge',
            safe= 'off').execute()

        import urllib.request as urllib
        def downloadImage(url, i):
            urllib.urlretrieve(url, 'imageDataSet/' + str(i))

        if not 'items' in res:
            print ('No result !!\nres is: {}'.format(res))
        else:
            for item in res['items']:
                print('{}:\n\t{}'.format(item['title'], item['link']))
                i += 1
                try:
                    downloadImage(item['link'], i)
                except:
                    print('Failed to download:', item['title'])


    curImage = convertImageToArray(3, size=(20, 20))
    plt.imshow(curImage)
    curImage = curImage.transpose(2, 0, 1)
    curImage = curImage[np.newaxis,:]
    scale, curImage = normalize(curImage)
    plt.show()
    reconstruct = theano.function([conv.x], conv.reconstructed)
    reconstructed = reconstruct(curImage)


    y = T.tensor4()
    err = T.mean(T.sqr(conv.reconstructed - y))

    (storage, rprop) = generateRpropUpdates(conv.params, err, init_size=0.001)
    (adamstorage, adam) = generateAdam(conv.params, err, alpha=0.001)
    (stor, rms) = generateRmsProp(conv.params, err, alpha=0.001)
    sgd = generateVanillaUpdates(conv.params, err, alpha=0.001)

    learn = theano.function([conv.x, y], err, updates=sgd)

    info = storage + conv.params

    try:
        loadParams(info, 'convae.npz')
    except Exception as e:
        print(e)

    curImage = curImage
    train_error = miniBatchLearning(
            curImage, curImage, -1, learn,
            verbose=True, epochs=1000)
    plt.imshow(np.squeeze(reconstructed).transpose(1, 2, 0))
    outImage = np.squeeze(scaleBack(reconstructed, scale)).transpose(1, 2, 0)
    print(outImage.max(), outImage.min())

    plt.figure()
    plt.imshow(conv.encode[0].w.get_value()[0,0,:,:], cmap='Greys')

    plt.show()
    plt.figure()
    plt.plot(np.arange(len(train_error)), train_error)
    plt.yscale('log')
    plt.show()
    # saveParams(info, 'convae')

    return

def KerasConvolutionDreamerTest():
    from keras.layers import Input
    from keras.models import Model
    from keras.layers import Convolution2D, MaxPooling2D, UpSampling2D
    from keras.optimizers import RMSprop

    def showImage(img):
        img = np.squeeze(img)
        plt.imshow(img.transpose(1, 2, 0))
        pass

    image = convertImageToArray(3, size=(800, 800))
    image2 = convertImageToArray(0, size=(800,800))
    image = image.transpose(2, 1, 0)
    image = image[np.newaxis,:]

    image = image.astype('float32') / 255
    showImage(image)
    plt.show()

    pic = Input(shape=(3, 800, 800))

    encoder = Convolution2D(16, 3, 3, activation='relu',
            border_mode='same')(pic)
    encoder = MaxPooling2D((2, 2), border_mode='same')(encoder)

    decoder = UpSampling2D((2, 2))(encoder)
    decoder = Convolution2D(3, 3, 3, activation='relu',
            border_mode='same')(decoder)

    model = Model(input=pic, output=decoder)

    rms = RMSprop(lr=0.001)
    model.compile(optimizer='adam', loss='mse')

    for i in range(10):
        model.fit(image, image, nb_epoch=20)
        generate = model.predict(image)
        for j in range(10):
            generate = model.predict(generate)
        plt.subplot(221)
        showImage(np.clip(model.predict(image), 0, 1))
        plt.subplot(222)
        showImage(np.clip(generate, 0, 1))
        plt.subplot(223)
        showImage(image)
        plt.show()


if __name__ == '__main__':
    HessianTester()
