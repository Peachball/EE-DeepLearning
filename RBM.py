from __future__ import print_function
import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt
import pickle
from DeepLearning import readMNISTData, readcv, miniBatchLearning, init_weights
from DeepLearning import saveParams, loadParams
from DCGAN import load_images

def sample(x, rng):
    return rng.binomial(size=x.shape, n=1, p=x, dtype=theano.config.floatX)

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

        self.params = [self.b, self.c, self.w]

    def getHidden(self, v):
        v = T.round(v)
        h = T.dot(v, self.w) + self.c
        h_sigmoid = T.nnet.sigmoid(h)
        h_bin = self.theano_rng.binomial(size=h.shape, n=1, p=h_sigmoid,
                dtype=theano.config.floatX)
        return [h, h_sigmoid, h_bin]

    def getVisible(self, h):
        h = T.round(h)
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
            return new_v, new_hid

        gibbsSample(persistent)
        ([chain, _], updates) = theano.scan(gibbsSample,
                outputs_info=[persistent, None], n_steps=k)

        chain_end = chain[-1]
        neg_cost = self.free_energy(chain_end)

        cost = T.mean(positive_cost) - T.mean(neg_cost)

        #Calculate gradient
        gparams = T.grad(cost, self.params, consider_constant=[chain_end])
        updates[persistent] = chain_end

        for g, p in zip(gparams, self.params):
            updates[p] = p - lr * g

        return cost, updates

    def gibbSample(self, startSample, k=1):
        sample = startSample
        for i in range(k):
            sample = self._getSample(sample)
        return sample


class ConvRBMLayer:
    def __init__(self, in_dim, out_filters, filt_size,
            theano_rng=T.shared_randomstreams.RandomStreams()):

        self.theano_rng = theano_rng
        self.c = init_weights(in_dim, init_type='uniform', scale=-1)
        self.b = init_weights(out_filters, init_type='uniform', scale=-1)
        self.w = init_weights((out_filters, in_dim) + filt_size,
                init_type='uniform', scale=-1)

        self.params = [self.w, self.b, self.c]

    def free_energy(self, v):
        visible_energy = -T.mean(v * self.c.dimshuffle('x', 0, 'x', 'x'),
                axis=(1, 2, 3))
        conv = T.nnet.conv.conv2d(v, self.w)
        hidden_energy = -T.nnet.softplus(conv +
                                    self.b.dimshuffle('x', 0, 'x',
                                        'x')).mean(axis=(1,2,3))

        return visible_energy + hidden_energy

    def getHidden(self, v):
        conv = T.nnet.conv2d(v, self.w, border_mode='half')
        act = conv + self.b.dimshuffle('x', 0, 'x', 'x')
        sigm = T.nnet.sigmoid(act)
        binary = self.theano_rng.binomial(size=sigm.shape, n=1, p=sigm,
                dtype=theano.config.floatX)
        return [act, sigm, binary]

    def getVisible(self, h):
        conv = T.nnet.conv2d(h, self.w.dimshuffle(1, 0, 2,
            3)[:,:,::-1,::-1], border_mode='half')
        act = conv + self.c.dimshuffle('x', 0, 'x', 'x')
        sigm = T.nnet.sigmoid(act)
        binary = self.theano_rng.binomial(size=sigm.shape, n=1, p=sigm,
                dtype=theano.config.floatX)
        return [act, sigm, binary]

    def getHiddenfromPool(self, p):
        pass

    def getPooling(self, v, pool_dim):
        hid_act, _, _ = self.getHidden(v)
        from theano.tensor.signal.pool import pool_2d
        pooled = pool_2d(T.exp(hid_act), pool_dim, ignore_border=True, mode='sum')
        summed = T.repeat(pooled, pool_dim[0], axis=2)
        summed = T.repeat(summed, pool_dim[1], axis=3)

        new_h_act = T.exp(hid_act) / (1 + summed)
        new_h_bin = self.theano_rng.binomial(size=new_h_act, n=1, p=new_h_act,
                dtype=theano.config.floatX)
        pool_prob = (1 - (1 / (1 + pooled)))
        pool_bin = self.theano_rng.binomial(size=pool_prob.shape, n=1,
                p=pool_prob, dtype=theano.config.floatX)

        return [new_h_act, new_h_bin, pool_prob, pool_bin]

    def pool_sample_vhv(self, v, pool):
        [h_act, h_bin, _, _] = self.getPooling(v, pool)
        [v_act, v_sig, v_bin] = self.getVisible(h_bin)
        return v_bin

    def sample_vhv(self, v, pool=None):
        [h_act, h_sig, h_bin] = self.getHidden(v)
        [v_act, v_sig, v_bin] = self.getVisible(h_bin)
        return v_sig, v_bin

    def mean_vhv(self, v, pool=None):
        v_tm1bin = self.theano_rng.binomial(size=v.shape, n=1, p=v,
                dtype=theano.config.floatX)
        [h_act, h_sig, h_bin] = self.getHidden(v_tm1bin)
        [v_act, v_sig, v_bin] = self.getVisible(h_bin)
        return v_sig

    def getCDUpdates(self, v, persistent, steps=1, pooling=None, sparsity=False):
        v_bin = self.theano_rng.binomial(size=v.shape, n=1, p=v,
                dtype=theano.config.floatX)
        visible_energy = self.free_energy(v_bin)

        def gibbsSample(prev_sample):
            v_sig, v_bin = self.sample_vhv(prev_sample)
            return v_bin, v_sig

        [samples, _], updates = theano.scan(fn=gibbsSample,
                outputs_info=[persistent, None], n_steps=steps)

        new_chain = samples[-1]
        upd = list(updates.items()) + [(persistent, new_chain)]
        energy = T.mean(self.free_energy(v)) -\
                T.mean(self.free_energy(new_chain))

        if not sparsity is None and not sparsity == False:
            energy = energy + sparsity * T.mean(T.sqr(self.w))

        grad = T.grad(energy, self.params, consider_constant=[new_chain])
        return upd, energy, grad

def RBMTester():
    images, labels = readMNISTData(10000)

    images = images.astype(theano.config.floatX)

    images = images / images.max()

    x = T.matrix('input')
    y = T.matrix()

    rbm = RBMLayer(784, 500, in_var=x)

    mse = T.mean(T.sqr(rbm.mean_vhv(x) - y))


    persistent = theano.shared(np.zeros((20,
        784)).astype(theano.config.floatX))
    _, updates = rbm.cost_updates(lr=0.001, persistent=persistent, k=15)


    learn = theano.function([rbm.x, y], mse, updates=updates,
            allow_input_downcast=True)

    generate = theano.function([rbm.x], rbm.mean_vhv(rbm.x),
            allow_input_downcast=True)

    sampled = np.zeros((1, 784)).astype('float32')
    miniBatchLearning(images, images, 20, learn, verbose=True, epochs=1)

    for i in range(100):
        sampled = generate(sampled)
        miniBatchLearning(images, images, 20, learn, verbose=True, epochs=1)
        plt.imshow(sampled[0].reshape(28, 28), cmap='Greys', interpolation='none')
        plt.show()

def ConvRBMTester():
    from keras.datasets import cifar10
    (X_train, _), _ = cifar10.load_data()
    X_train = X_train.astype('float32') / 255
    X = T.tensor4()
    x_mnist, _ = readMNISTData(10000)
    x_mnist = x_mnist.reshape(-1, 1, 28, 28) / 255
    print(x_mnist.shape)
    rbm = ConvRBMLayer(3, 24, (9, 9))
    _, _, hid_bin = rbm.getHidden(X)
    energy = rbm.free_energy(X)
    persistent = theano.shared(np.random.rand(2, 3, 32, 32).astype(theano.config.floatX))
    upd, error, grad = rbm.getCDUpdates(X, persistent, steps=2, sparsity=False)

    origupd = upd
    upd += [(p, p - 0.3 * g) for p, g in zip(rbm.params, grad)]

    learn = theano.function([X], error, updates=upd, allow_input_downcast=True)
    generate = theano.function([X], rbm.mean_vhv(X),
            allow_input_downcast=True)

    gradient = theano.function([X], [error, T.mean(T.abs_(grad[0]))], updates=origupd, allow_input_downcast=True)
    # try:
        # loadParams(rbm.params, 'convrbm.npz')
    # except:
        # print("failed to load params")
    f = open('convrbmerr.txt', 'a')
    iteration = 0
    batchSize = 10
    counter = 0
    saveParams(rbm.params, 'convrbm.npz')
    plt.ion()
    for i in range(10000):
        index = 0#i * 20 % 10000
        print(learn(X_train[index:index+2]))
        # print(gradient(X_train[index:index+20]))
        # print(np.mean(np.abs(rbm.w.get_value())))
        counter += 1
        img = generate(persistent.get_value())
        plt.subplot(2, 2, 1)
        plt.imshow(img[0].transpose(1, 2, 0) * 255)
        plt.subplot(2, 2, 2)
        gen_i = generate(X_train[index:index+1])
        plt.imshow(gen_i[0].transpose(1, 2, 0) * 255)
        plt.subplot(2, 2, 3)
        plt.imshow(X_train[index].transpose(1, 2, 0) * 255)
        plt.subplot(2, 2, 4)
        plt.imshow(X_train[index + 1].transpose(1, 2, 0) * 255)
        plt.pause(0.05)
        if counter % 50 == 0:
            saveParams(rbm.params, 'convrbm.npz')

def basic_ConvRBMTester():
    X = T.tensor4()
    rbm = ConvRBMLayer(1, 24, (15, 15))
    persistent = theano.shared(np.random.rand(9, 1, 30,
        30).astype(theano.config.floatX))
    upd, energy, grad = rbm.getCDUpdates(X, persistent, steps=1)

    gradUpdates = [(p, p - 0.1 * g) for p, g in zip(rbm.params, grad)]
    updates = upd + gradUpdates

    x_mnist, _ = readMNISTData(10)
    x_mnist = x_mnist.reshape(-1, 1, 28, 28).astype('float32') / 255
    img = np.ones((1, 1, 30, 30)).astype(theano.config.floatX)
    img[0,0,0,0] = 0
    learn = theano.function([X], energy, updates=updates,
            allow_input_downcast=True)
    generate = theano.function([X], rbm.mean_vhv(X), allow_input_downcast=True)

    plt.ion()
    for i in range(10000):
        print(learn(x_mnist[0:1]))
        if i % 100 == 0:
            plt.figure(1)
            for i in range(9):
                plt.subplot(3, 3, i+1)
                plt.imshow(persistent.get_value()[i, 0] * 255, cmap='Greys')
            plt.figure(2)
            plt.imshow(generate(x_mnist[0:1])[0,0] * 255, cmap='Greys')
            plt.pause(0.05)


    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.imshow(persistent.get_value()[i, 0] * 255, cmap='Greys')
    plt.figure()
    plt.imshow(x_mnist[1, 0] * 255, cmap='Greys')
    plt.show()

if __name__== '__main__':
    ConvRBMTester()
