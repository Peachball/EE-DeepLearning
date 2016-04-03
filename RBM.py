from __future__ import print_function
import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt
from DeepLearning import readMNISTData, readcv

class RBMLayer:
    def __init__(self, in_size, out_size, visible_type='det', hidden_type='det',
            distribution=T.nnet.sigmoid, init_size=0.1):
        self.in_size = in_size
        self.out_size = out_size
        
        self.visible = T.vector('Visible')
        '''theano.shared(value=(np.random.rand(in_size) - 0.5) *
                init_size, name='Visible RBM Units').astype(theano.config.floatX)
        '''
        self.hidden = T.vector('Hidden')
        '''theano.shared(value=(np.random.rand(out_size) - 0.5) *
                init_size, name='Hidden RBM Units').astype(theano.config.floatX)
        '''

        self.b = theano.shared(value=(np.random.rand(in_size) - 0.5) *
                init_size, name='Visible Biases').astype(theano.config.floatX)
        self.c = theano.shared(value=(np.random.rand(out_size) - 0.5) * init_size,
            name='Hidden Biases').astype(theano.config.floatX)

        self.w = theano.shared(value=(np.random.rand(in_size, out_size) - 0.5) * init_size,
        name='RBM Weights').astype(theano.config.floatX)

        self.h = distribution(T.dot(self.visible.T, self.w) + self.b)
        self.p = distribution(T.dot(self.hidden.T, self.w.T) + self.c)

    def CDUpdates(self, alpha):
        updates = []

def RBMTester():
    rbm = RBMLayer(784, 100)

if __name__== '__main__':
    RBMTester()

