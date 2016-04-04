from __future__ import print_function
import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt
from DeepLearning import readMNISTData, readcv

class RBMLayer:
    def __init__(self, in_size, out_size, visible_type='det', hidden_type='det',
            distribution=T.nnet.sigmoid, init_size=0.1, in_var=T.matrix('input')):
        self.in_size = in_size
        self.out_size = out_size
        self.distribution = distribution
        self.in_var = in_var
        
        self.b = theano.shared(value=(np.random.rand(in_size) - 0.5) *
                init_size, name='Visible Biases').astype(theano.config.floatX)
        self.c = theano.shared(value=(np.random.rand(out_size) - 0.5) * init_size,
            name='Hidden Biases').astype(theano.config.floatX)

        self.w = theano.shared(value=(np.random.rand(in_size, out_size) - 0.5) * init_size,
        name='RBM Weights').astype(theano.config.floatX)

        hidden = distribution(T.dot(in_var, self.w) + self.c)
        self.out = distribution(T.dot(hidden, self.w.T) + self.b)
        self._getSample = theano.function([in_var], self.out)

    def CDUpdates(self, x, alpha):

        def persistentCD():
            pass
        updates = []

        negativeSample = T.matrix('neg sample')
        #Weight updates
        hidden = self.distribution(T.dot(x, self.w) + self.b)
        negHidden = self.distribution(T.dot(negativeSample, self.w) + self.b)
        weightGrad = T.dot(x, hidden) / x.shape[0] - T.dot(negativeSample, negHidden) / \
            negativeSample.shape[0]
        updates.append((self.w, self.w + alpha * (weightGrad)))

        #Bias updates
        updates.append((self.b, self.b + alpha * (x - negativeSample)))
        updates.append((self.c, self.c + alpha * (hidden - negHidden)))

        return updates

    def gibbSample(self, startSample, k=1):
        sample = startSample
        for i in range(k):
            sample = self._getSample(sample)

        return sample

def RBMTester():
    images, labels = readMNISTData(10)

    images = images / images.max()

    print(images.shape)

    rbm = RBMLayer(784, 600)

    y = T.matrix()
    mse = T.mean(T.sqr(rbm.out - y))

    negSample = theano.shared(value=rbm.gibbSample(images)).astype(theano.config.floatX)
    updates = rbm.CDUpdates(rbm.in_var, 0.01)

    learn = theano.function([rbm.in_var, y], mse, updates=updates)

    print(learn(images, images))
    
    sampled = rbm.gibbSample(images)

    plt.imshow(sampled[0].reshape(28, 28), cmap='Greys', interpolation='none')
    plt.show()
if __name__== '__main__':
    RBMTester()

