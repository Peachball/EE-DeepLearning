from __future__ import print_function
import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt
import pickle
from DeepLearning import readMNISTData, readcv, miniBatchLearning

class RBMLayer:
    def __init__(self, in_size, hidden_size, visible_type='det', hidden_type='det',
            distribution=T.nnet.sigmoid, init_size=0.1, in_var=T.matrix('input'),
            persistent_updatesize=100, theano_rng=None):
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.distribution = distribution
        self.in_var = in_var
        self.x = in_var

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

        self.params = [self.b, self.c, self.w]

    def getHidden(self, v):
        h = T.dot(v, self.w) + self.c
        h_sigmoid = T.nnet.sigmoid(h)
        h_bin = self.theano_rng.binomial(size=h.shape, n=1, p=h_sigmoid,
                dtype=theano.config.floatX)
        return [h, h_sigmoid, h_bin]

    def getVisible(self, h):
        v = T.dot(h, self.w.T) + self.b
        v_sigmoid = T.nnet.sigmoid(v)
        v_bin = self.theano_rng.binomial(size=v.shape, n=1, p=v_sigmoid,
                dtype=theano.config.floatX)
        return [v, v_sigmoid, v_bin]

    def sample_vhv(self, v):
        [h, h_sig, h_bin] = self.getHidden(v)
        [v1, v1_sig, v1_bin] = self.getVisible(h_bin)
        return v1_bin

    def mean_vhv(self, v):
        [h, h_sig, h_bin] = self.getHidden(v)
        [v, v_sig, v_bin] = self.getVisible(h_bin)
        return v_sig

    #Determine F(v, h) using only v
    def free_energy(self, v):
        bias = T.dot(v, self.b)

        # [_, hidden, _] = self.getHidden(v)
        hidden = T.dot(v, self.w) + self.c
        not_bias = T.sum(T.log(1 + T.exp(hidden)), axis=1)

        return -bias - not_bias


    def cost_updates(self, lr=0.1, persistent=None, k=1):
        positive_cost = self.free_energy(self.x)
        lr = theano.shared(np.array(lr).astype(theano.config.floatX))

        if not persistent:
            persistent = theano.shared(np.zeros((1,
                self.in_size)).astype(theano.config.floatX))

        def gibbsSample(prev_chain):
            _, _, new_hid = self.getHidden(prev_chain)
            _, _, new_v = self.getVisible(new_hid)
            return new_hid, new_v

        gibbsSample(persistent)
        ([_, chain], updates) = theano.scan(gibbsSample,
                outputs_info=[None, persistent], n_steps=k)

        chain_end = chain[-1]
        neg_cost = self.free_energy(chain_end)

        cost = T.mean(positive_cost) - T.mean(neg_cost)

        #Calculate gradient
        gparams = T.grad(cost, self.params, consider_constant=[chain_end])
        gradUpdates = []
        gradUpdates.append((persistent, chain_end))

        for g, p in zip(gparams, self.params):
            gradUpdates.append((p, p - lr * g))

        return cost, gradUpdates


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

class RNNRBMLayer:
    def __init__(self, in_size, v_size, hidden_size):
        pass

def RBMTester():
    theano.config.floatX='float32'
    images, labels = readMNISTData(1000)

    images = images.astype(theano.config.floatX)

    images = images / images.max()

    x = T.matrix()
    y = T.matrix()

    rbm = RBMLayer(784, 600, in_var=x)

    mse = T.mean(T.sqr(rbm.sample_vhv(x) - y))

    negSample = theano.shared(value=rbm.gibbSample(images)).astype(theano.config.floatX)

    persistent = theano.shared(np.zeros((10,
        784)).astype(theano.config.floatX))
    _, updates = rbm.cost_updates(lr=0.01, persistent=persistent)


    learn = theano.function([rbm.x, y], mse, updates=updates,
            allow_input_downcast=True)

    miniBatchLearning(images, images, 100, learn, verbose=True, epochs=100)

    sampled = rbm.gibbSample(images)

    for i in range(10):
        plt.imshow(sampled[i].reshape(28, 28), cmap='Greys', interpolation='none')
        plt.show()


if __name__== '__main__':
    RBMTester()
