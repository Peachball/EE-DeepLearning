from __future__ import print_function
import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt
import pickle
from DeepLearning import readMNISTData, readcv

class RBMLayer:
    def __init__(self, in_size, hidden_size, visible_type='det', hidden_type='det',
            distribution=T.nnet.sigmoid, init_size=0.1, in_var=T.matrix('input'),
            persistent_updatesize=100, theano_rng=None):
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.distribution = distribution
        self.in_var = in_var

        if theano_rng == None:
            theano_rng = T.shared_randomstreams.RandomStreams()

        self.theano_rng = theano_rng
        self.b = theano.shared(value=np.random.uniform(low=-init_size,
            high=init_size, size=(in_size)).astype(theano.config.floatX),
            name='Visible Biases')
        self.c = theano.shared(value=np.random.uniform(low=-init_size,
            high=init_size, size=(hidden_size)).astype(theano.config.floatX),
            name='Hidden Biases')

        self.w = theano.shared(value=np.random.uniform(low=-init_size,
            high=init_size, size=(in_size, hidden_size))
            .astype(theano.config.floatX), name="RBM Weights")

        hidden = distribution(T.dot(in_var, self.w) + self.c)
        self.out = distribution(T.dot(hidden, self.w.T) + self.b)
        self._getSample = theano.function([in_var], self.out)

        self.persistentCD = theano.shared(
                value=np.random.uniform(low=-init_size, high=init_size,
                    size=(persistent_updatesize, in_size))
                .astype(theano.config.floatX),
                name='Persistent CD')

    def getHidden(self, v):
        h = T.dot(v, self.w) + self.c
        h_sigmoid = T.nnet.sigmoid(h)
        h_bin = self.theano_rng.binomial(size=h.shape, n=1, p=h_sigmoid)
        return [h, h_sigmoid, h_bin]

    def getVisible(self, h):
        v = T.dot(h, self.w.T) + self.b
        v_sigmoid = T.nnet.sigmoid(v)
        v_bin = self.theano_rng.binomial(size=v.shape, n=1, p=v_sigmoid)
        return [v, v_sigmoid, v_bin]

    def sample_vhv(self, v):
        [h, h_sig, h_bin] = self.getHidden(v)
        [v1, v1_sig, v1_bin] = self.getVisible(h_bin)
        return v1_bin

    def free_energy(self, v):
        bias = T.dot(v, self.b)

        [_, hidden, _] = getHidden(v)
        not_bias = T.sum(T.log(1 + T.exp(hidden)), axis=1)

        return -bias - not_bias


    def CDUpdates(self, x, alpha):
        negativeSample = self.persistentCD
        updates = []

        #Weight updates
        hidden = self.distribution(T.dot(x, self.w) + self.c)
        negHidden = self.distribution(T.dot(negativeSample, self.w) + self.c)
        weightGrad = (T.dot(x.T, hidden))/x.shape[0] - T.dot(negativeSample.T, negHidden)/negHidden.shape[0]
        updates.append((self.w, self.w + alpha * (weightGrad)))

        #Bias updates
        updates.append((self.b, self.b + T.mean(alpha * (x - negativeSample), axis=0)))
        updates.append((self.c, (self.c + T.mean(alpha * (hidden - negHidden), axis=0))))

        #Persistent Contrastive Divergence
        newCD_h = self.distribution(T.dot(self.persistentCD, self.w) + self.c)
        newCD_v = self.distribution(T.dot(newCD_h, self.w.T) + self.b)
        updates.append((self.persistentCD, newCD_v))
        return updates

    def getHidden(self, x):
        return self.distribution(T.dot(x, self.w) + self.b)

    def gibbSample(self, startSample, k=1):
        sample = startSample
        for i in range(k):
            sample = self._getSample(sample)
        return sample

    def miniBatch(self, learnFunction, x, epochs=1, verbose=False):
        train_error = []
        batchSize = self.persistentCD.get_value().shape[0]

        for i in range(epochs):
            for j in range(0, x.shape[0] - batchSize + 1, batchSize):
                learnx = x[j:(j+batchSize)]
                error = learnFunction(learnx, learnx)
                if verbose: print(error, "Epoch:", i + j/x.shape[0])
                train_error.append(error)
        return train_error

def RBMTester():
    images, labels = readMNISTData(1000)

    images = images / images.max()


    rbm = RBMLayer(784, 600)

    y = T.matrix()
    mse = T.mean(T.sqr(rbm.out - y))

    negSample = theano.shared(value=rbm.gibbSample(images)).astype(theano.config.floatX)
    updates = rbm.CDUpdates(rbm.in_var, 0.01)


    learn = theano.function([rbm.in_var, y], mse, updates=updates,
            allow_input_downcast=True)


    rbm.miniBatch(learn, images, verbose=True, epochs=100)
    sampled = rbm.gibbSample(images)

    for i in range(10):
        plt.imshow(sampled[i].reshape(28, 28), cmap='Greys', interpolation='none')
        plt.show()


if __name__== '__main__':
    RBMTester()
