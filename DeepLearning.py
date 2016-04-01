from __future__ import print_function
from collections import OrderedDict
import theano.tensor as T
import theano
from theano import config
import numpy as np
import struct
import matplotlib.pyplot as plt

theano.config.floatX = 'float64'

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
        print('\r Read {}/{}'.format(i+1, length), end="")
    print('Done reading')
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
        print('\r Read {}/{}'.format(i+1, length), end="")
        
    print ('Done Reading')
    return (np.array(imgs), np.array(lbls))


class Layer:
    def __init__(self, in_size, out_size, layer_type='sigmoid', in_var=None,
            init_size=0.1):
        self.in_size = in_size
        self.out_size = out_size
        if not layer_type in ['sigmoid', 'tanh', 'lstm', 'rnn', 'linear']:
            raise Exception('Layer type is invalid: ' + str(layer_type))
        
        if in_var==None:
            x = T.matrix('Input')
        else:
            x = in_var
        
        if layer_type in ['sigmoid', 'tanh', 'linear']:
            self.w = theano.shared((np.random.rand(in_size, out_size) - 0.5) *
                    init_size).astype(theano.config.floatX)
            self.b = theano.shared((np.random.rand(out_size) - 0.5) *
                    init_size).astype(theano.config.floatX)
            if layer_type == 'sigmoid':
                self.out = T.nnet.sigmoid( T.dot(x, self.w) + self.b )
            if layer_type == 'tanh':
                self.out = T.tanh( T.dot(x, self.w) + self.b )
            if layer_type == 'linear':
                self.out = x * self.w + self.b
            self.params = [self.w, self.b]

def generateMomentumUpdates(params, momentum, alpha, error):
    grad = []
    for p in params:
        grad.append(T.grad(error, p))
    mparams = [theano.shared(np.zeros(p.get_value().shape)).astype(theano.config.floatX) for p in params]
    gradUpdates = OrderedDict((p, p - g) for p, g in zip(params,mparams))

    gradUpdates.update(OrderedDict((m, momentum * m + alpha * g) for m, g in
        zip(mparams, grad)))
    return ([gradUpdates, mparams], gradUpdates)

def generateRpropUpdates(params, error, init_size=1):
    prevw = []
    deltaw = []
    updates = []
    gradients = []
    #initalize stuff
    for p in params:
        prevw.append(theano.shared(np.zeros(p.get_value().shape)).astype(config.floatX))
        deltaw.append(theano.shared(init_size *  np.ones(p.get_value().shape)).astype(config.floatX))

    for p, dw, pw in zip(params, deltaw, prevw):
        try:
            gradients.append(T.grad(error, p))
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

    return (storage, updates)

class AutoEncoder:
    def __init__(self, *dim, **kwargs):
        momentum = kwargs.get('momentum', 0)
        alpha = kwargs.get('alpha', 0.01)
        rprop = kwargs.get('rprop', False)
        init_size = kwargs.get('init_size', 1)
        verbose = kwargs.get('verbose', False)
        layers = []

        self.x = kwargs.get('in_var', T.matrix('Generalized Input'))
        layers.append(Layer(dim[0], dim[1], in_var=self.x, init_size=init_size))
        for i in range(1, len(dim) - 1):
            layers.append(Layer(dim[i], dim[i+1], in_var=layers[-1].out, init_size=init_size))

        self.out = layers[-1].out
        decoder = []

        decoder.append(Layer(dim[-1], dim[-2], in_var=self.out, init_size=init_size))
        for i in range(2, len(dim)):
            decoder.append(Layer(dim[-i], dim[-i-1], in_var=decoder[-1].out, init_size=init_size))

        self.reconstructed = decoder[-1].out

        self.encode = layers
        self.decode = decoder

        if verbose: print(len(layers), len(decoder))
        self.layers = self.encode + self.decode

        self.params =[]

        for l in self.layers:
            self.params = self.params + l.params


        print('Finished initialization')

def matplotlibQuitter(event):
    if event.key == 'q':
        quit()

def AETester():
    images, labels = readMNISTData(10000)
    xcv, ycv = readcv(1)
    ae = AutoEncoder(784, 600, 500, init_size=10)

#    images = images / images.max()

    genImage = theano.function([ae.x], ae.reconstructed, mode='DebugMode')

    y = T.matrix('correct output')
    yl = T.matrix('Correct labels')
    mse = T.mean(T.sqr(y - ae.reconstructed))
    
    crossEntrop = -T.mean(yl * T.log(ae.out) + (1 - yl) * T.log(1 - ae.out))

    (momentumStorage, updates) = generateMomentumUpdates(ae.params, 0.9, 10, mse)
    (rprop, rpropupdates) = generateRpropUpdates(ae.params, mse, init_size=0.01)

    learn = theano.function([ae.x, y], mse, updates=rpropupdates)
    train_error = []
    iterations = 1000
    for i in range(iterations):
        e = learn(images, images)
        train_error.append(e)
        print(e)

    plt.plot(np.arange(0, iterations), train_error)
#    plt.ylim(0, plt.ylim()[1])
    plt.show()

    for i in range(len(images)):
        generated_image = genImage(images[i].reshape(1, 784))
        plt.imshow(generated_image.reshape(28, 28), cmap='Greys')
        plt.figure()
        plt.imshow(images[i].reshape(28, 28), cmap='Greys')
        plt.show()

if __name__ == '__main__':
    AETester()
