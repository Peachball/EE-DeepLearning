from __future__ import print_function
import theano.tensor as T
import theano
import numpy as np

def readMNISTData(length=10000):
    images = open('../train-images.idx3-ubyte', 'rb')
    labels = open('../train-labels.idx1-ubyte', 'rb')
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
    images = open('../t10k-images.idx3-ubyte', 'rb')
    labels = open('../t10k-labels.idx1-ubyte', 'rb')
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
                self.out = T.nnet.sigmoid( x * self.w + self.b )
            if layer_type == 'tanh':
                self.out = T.tanh( x * self.w + self.b )
            if layer_type == 'linear':
                self.out = x * self.w + self.b
            self.params = [self.w, self.b]

def generateMomentumUpdates(params, momentum, alpha):
    mparams = [theano.shared(np.zeros(g.eval().shape)).astype(config.floatX) for g in grad]
    gradUpdates = OrderedDict((p, p - g) for p, g in zip(self.params, self.mparams))

    gradUpdates.update(OrderedDict((m, self.momentum * m + self.alpha * g) for m, g in
        zip(mparams, grad)))
    return (gradUpdates, mparams)

def generateRpropUpdates(params):


class AutoEncoder:
    def __init__(self, *dim, **kwargs):
        momentum = kwargs.get('momentum', 0)
        alpha = kwargs.get('alpha', 0.01)
        rprop = kwargs.get('rprop', False)
