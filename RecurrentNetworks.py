import numpy as np
import theano
import theano.tensor as T
from theano import config
from DeepLearning import *
import matplotlib.pyplot as plt
import time

class RecurrentLayer:

    def __init__(self, in_size, out_size, in_var=T.matrix('input'),
            init_size=0.01, verbose=False, hidden_size=None):
        x = in_var
        self.x = x
        if hidden_size == None:
            hidden_size = max(in_size, out_size)

        w_io = theano.shared(value=(np.random.rand(in_size, out_size) - 0.5) *
                init_size).astype(theano.config.floatX)
        b_io = theano.shared(value=(np.random.rand(out_size) - 0.5) *
                init_size).astype(theano.config.floatX)

        w_ih = theano.shared(value=(np.random.rand(in_size, hidden_size) -
            0.5) * init_size).astype(theano.config.floatX)
        b_ih = theano.shared(value=(np.random.rand(hidden_size) - 0.5) *
                init_size).astype(theano.config.floatX)

        w_hh = theano.shared(value=np.random.uniform(low=-init_size, high=init_size,
            size=(hidden_size, hidden_size)))
        b_hh = theano.shared(value=np.random.uniform(low=-init_size, high=init_size,
            size=(hidden_size)))

        w_ho = theano.shared(value=np.random.uniform(low=-init_size, high=init_size,
            size=(hidden_size, out_size)))
        b_ho = theano.shared(value=np.random.uniform(low=-init_size, high=init_size,
            size=(out_size)))
        self.hidden = theano.shared(
                value=np.zeros(hidden_size)).astype(theano.config.floatX)

        def recurrence(x, h_tm1):
            h_t = T.dot(x, w_ih) + b_ih + T.dot(h_tm1, w_hh) + b_hh
            out = T.dot(x, w_io) + b_io + T.dot(h_tm1, w_ho) + b_ho
            return [out, h_t]

        ([out, hidden], updates) = theano.scan(recurrence,
                sequences=x, outputs_info=[None, T.matrix("hidden")], n_steps=x.shape[0])

        self.out = out
        self.updates = updates
        self.hidden = hidden[-1]

        self.params = [w_ih, b_ih, w_hh, b_hh, w_io, b_io, w_ho, b_ho]

class RNN:

    def __init__(self, *dim, **kwargs):
        init_size = kwargs.get('init_size', 0.01)
        verbose = kwargs.get('verbose', False)

        x = T.matrix("input")
        layers = []
        layers.append(RecurrentLayer(dim[0], dim[1], init_size = init_size, in_var=x))

        for i in range(1, len(dim) -1):
            layers.append(RecurrentLayer(dim[i], dim[i+1], init_size=init_size,
                in_var=layers[-1].out))

        self.out = layers[-1].out


class LSTMLayer:
    '''
        This assumes that in this recurrent net, there is a corresponding output to each input
    '''
    def __init__(self, in_size, out_size, cell_size=None, init_size=0.01,
            out_type='sigmoid', in_var=None, out_var=None, verbose=False):
        if cell_size == None:
            cell_size = max(in_size, out_size)
        self.in_size = in_size
        self.out_size = out_size
        self.cell_size = cell_size
        self.C = theano.shared(value=np.zeros((1, cell_size)), name='LongTerm')
        self.h = theano.shared(value=np.zeros((1, out_size)), name='Previous Prediction')
        if in_var == None:
            x = T.matrix(name='input example')
        else:
            x = in_var
        if verbose:
            print('Constants have been initalized')

        #Forget gate
        self.W_xf = theano.shared(
                value=np.random.uniform(low=-init_size, high=init_size,
                    size=(in_size, cell_size)).astype(theano.config.floatX))
        self.b_f= theano.shared(
                value=np.random.uniform(low=-init_size, high=init_size,
                    size=(1, cell_size)).astype(theano.config.floatX))
        self.W_cf = theano.shared(
                value=np.random.uniform(low=-init_size, high=init_size,
                    size=(cell_size, cell_size)).astype(theano.config.floatX))
        self.W_hf = theano.shared(
                value=np.random.uniform(low=-init_size, high=init_size,
                    size=(out_size, cell_size)).astype(theano.config.floatX))

#        forget = T.nnet.sigmoid(T.dot(self.h, self.W_hf) + T.dot(self.C, self.W_cf) + T.dot(self.x, self.W_xf) + self.b_f)

        #Memories
        self.W_hm = theano.shared(
                value=np.random.uniform(low=-init_size, high=init_size,
                    size=(out_size, cell_size)).astype(theano.config.floatX))
        self.W_xm = theano.shared(
                value=np.random.uniform(low=-init_size, high=init_size,
                    size=(in_size, cell_size)).astype(theano.config.floatX))
        self.b_m = theano.shared(
                value=np.random.uniform(low=-init_size, high=init_size,
                    size=(1, cell_size)).astype(theano.config.floatX))

#        memories = T.tanh(T.dot(self.h, self.W_hm) + T.dot(self.x, self.W_xm) + self.b_m)
        #Remember Gate
        self.W_hr = theano.shared(
                value=np.random.uniform(low=-init_size, high=init_size,
                    size=(out_size, cell_size)).astype(theano.config.floatX))
        self.W_xr = theano.shared(
                value=np.random.uniform(low=-init_size, high=init_size,
                    size=(in_size, cell_size)).astype(theano.config.floatX))
        self.W_cr = theano.shared(
                value=np.random.uniform(low=-init_size, high=init_size,
                    size=(cell_size, cell_size)).astype(theano.config.floatX))
        self.b_r = theano.shared(
                value=np.random.uniform(low=-init_size, high=init_size,
                    size=(1, cell_size)).astype(theano.config.floatX))

#        remember = T.nnet.sigmoid(T.dot(self.h, self.W_hr) + T.dot(self.C, self.W_cr) + T.dot(self.x, self.W_xr) + self.b_r)

        #Output
        self.W_co = theano.shared(
                value=np.random.uniform(low=-init_size, high=init_size,
                    size=(cell_size, out_size)).astype(theano.config.floatX))
        self.W_ho = theano.shared(
                value=np.random.uniform(low=-init_size, high=init_size,
                    size=(out_size, out_size)).astype(theano.config.floatX))
        self.W_xo = theano.shared(
                value=np.random.uniform(low=-init_size, high=init_size,
                    size=(in_size, out_size)).astype(theano.config.floatX))
        self.b_o = theano.shared(
                value=np.random.uniform(low=-init_size, high=init_size,
                    size=(1, out_size)).astype(theano.config.floatX))

        #Hidden
        self.W_ch = theano.shared(
                value=np.random.uniform(low=-init_size, high=init_size,
                    size=(cell_size, out_size)).astype(theano.config.floatX))
        self.W_hh = theano.shared(
                value=np.random.uniform(low=-init_size, high=init_size,
                    size=(out_size, out_size)).astype(theano.config.floatX))
        self.W_xh = theano.shared(
                value=np.random.uniform(low=-init_size, high=init_size,
                    size=(in_size, out_size)).astype(theano.config.floatX))
        self.b_h = theano.shared(
                value=np.random.uniform(low=-init_size, high=init_size,
                    size=(1, out_size)).astype(theano.config.floatX))

        self.params = [self.W_xf, self.W_hf, self.W_cf, self.b_f, self.W_hm, self.W_xm, self.b_m,
                self.W_hr, self.W_cr, self.W_xr, self.b_r, self.W_co, self.W_ho,
                self.W_xo, self.b_o, self.W_ch, self.W_hh, self.W_xh, self.b_h]
        if verbose:
            print('Weights have been initalized')

        def recurrence(x, h_tm1, c_tm1):
            rem = T.nnet.sigmoid(T.dot( h_tm1, self.W_hr) + T.dot( c_tm1 , self.W_cr) + T.dot( x, self.W_xr) + self.b_r)
            mem = T.tanh(T.dot( h_tm1, self.W_hm) + T.dot( x, self.W_xm) + self.b_m)
            forget = T.nnet.sigmoid(T.dot( h_tm1, self.W_hf) + T.dot( c_tm1, self.W_cf) + T.dot( x, self.W_xf) + self.b_f)

            z = T.dot(c_tm1 , self.W_co) + T.dot( h_tm1 , self.W_ho) + T.dot(x, self.W_xo) + self.b_o
            h_t = T.nnet.sigmoid(T.dot(c_tm1, self.W_ch) + T.dot(h_tm1, self.W_hh) + T.dot(x,
                self.W_xh) + self.b_h)
            if out_type=='sigmoid':
                out = T.nnet.sigmoid(z)
            elif out_type=='linear':
                out = z
            elif out_type=='softmax':
                out = T.nnet.softmax(z)

            c_t = self.C * forget + rem * mem
            return [z, h_t, c_t]

        ([actualOutput, hidden, cell_state], _) = theano.scan(fn=recurrence,
                sequences=x,
                outputs_info=[None, self.h, self.C],
                n_steps=x.shape[0])
        if verbose:
            print('Recurrence has been set up')

        self.hidden = hidden
        self.cell_state = cell_state
        output = actualOutput.reshape((actualOutput.shape[0], actualOutput.shape[2]))
        self.out = output

    def reset(self):
        self.C.set_value(np.zeros(self.C.shape.eval()))
        self.h.set_value(np.zeros(self.h.shape.eval()))

class LSTM():

    def __init__(self, *dim, **kwargs):
        out_type = kwargs.get('out_type', 'sigmoid')
        self.layers = []
        verbose = kwargs.get('verbose', False)
        init_size = kwargs.get('init_size', 0.01)
        x = kwargs.get("in_var", T.matrix('Input'))
        y = T.matrix('Output')
        self.x = x
        self.y = y
        init_size = kwargs.get('init_size', 1e-10)
        self.layers.append(LSTMLayer(dim[0], dim[1], in_var=x, verbose=False))
        for i in range(1, len(dim) - 1):
            self.layers.append(LSTMLayer(dim[i], dim[i+1], in_var=self.layers[-1].out, init_size=init_size, out_type='sigmoid'))
            if i == len(dim)-2:
                self.layers[-1] = LSTMLayer(dim[i], dim[i+1], in_var=self.layers[-2].out, out_type=out_type, init_size=init_size)


        if verbose:
            print('Number of layers:' + str(len(self.layers)))
            print('Finished with initialization of layers -> Defining prediction')

        #Defining updates for all layers:
        layerUpdates = []
        for l in self.layers:
            layerUpdates.append((l.C, l.cell_state[-1]))
            layerUpdates.append((l.h, l.hidden[-1]))

        #Define prediction:
        prediction = self.layers[-1].out
        self.predict = theano.function([x], prediction, updates=layerUpdates)
        self.out = prediction

        if verbose:
            print('Defining error')
        #Define Error
        if out_type=='sigmoid':
            self.error = -T.mean((y)*T.log(T.clip(prediction, 1e-9, 1-1e-9)) +
                    (1-y)*T.log(1-T.clip(prediction, 1e-9, 1-1e-9)))
        elif out_type=='linear':
            self.error = T.mean(T.sqr(y - prediction))

        if verbose:
            print('Wrapping paramaters')
        #Define paramater list
        self.params = []
        for i in self.layers:
            self.params = self.params + i.params


    def resetGrad(self):
        for sumGrad in self.summedGradients:
            sumGrad.set_value(np.zeros(sumGrad.shape))

    def reset(self):
        for l in self.layers:
            l.reset()
        return

    def load_npz(self, npz):
        j = {}
        for i in npz:
            j[int(i.replace('arr_', ''))] = npz[i]
        return j

    def get_numpy(self):
        num = []
        for p in self.params:
            num.append(p.get_value())
        return num

    def save(self, f):
        arr = self.get_numpy()
        np.savez(f, *arr)

    def load(self, f):
        params = self.load_npz(np.load(f))
        for p, n in zip(self.params, params):
            p.set_value(params[n])


def miniRecurrentLearning(x, y, batchSize, learn, predict, verbose=False,
        epochs=1, miniepochs=10):
    train_error = []
    if batchSize <= 0: batchSize = x.shape[0]
    for j in range(epochs):
        print(batchSize)
        for batch in range(0, x.shape[0], batchSize):
            train_error = train_error + miniBatchLearning(x[batch:batch+batchSize],
                    y[batch:batch+batchSize], -1, learn, verbose=verbose, 
                    epochs=miniepochs)
        predict(x[batch:batchSize])
    return train_error


def LSTMTest():
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    x = x.reshape(x.shape[0], 1)
    y = y.reshape(y.shape[0], 1)
    lstm = LSTM(1, 10, 1, verbose=True)
    sin_fig, ax = plt.subplots()
    ax.plot(x, y, label="Sine curve")

    er_fig = plt.figure()
    er_dat = er_fig.add_subplot(111)

    target = T.matrix("target")
    error = T.mean(T.sqr(lstm.out - target))

    print("Computing gradients...")
    sgd = generateVanillaUpdates(lstm.params, error, alpha=0.01)
    (adamstorage, adam) = generateAdam(lstm.params, error, alpha=0.01)
    (rp, rprop) = generateRpropUpdates(lstm.params, error, 0.1)
    (rmstorage, rms) = generateRmsProp(lstm.params, error, 0.01, decay=0.9)
    (stor, adadelta) = generateAdadelta(lstm.params, error, alpha=1, decay=0.9)
    (ada, adagrad) = generateAdagrad(lstm.params, error, alpha=0.01)

    updateRules = [(sgd, "SGD"), (adam, "Adam"), (rms, "RMS"), (adadelta, "Adadelta"),
            (adagrad, "Adagrad"), (rprop, "RProp")]

    def train(learn, epochs=100, verbose=False):
        train_error = []
        for i in range(epochs):
            error = learn(x, y)
            train_error.append(error)
            if verbose: print(error)
        return train_error

    for u in updateRules:
        reset(lstm.params)
        print("Compiling", u[1], "function")
        learn = theano.function([lstm.x, target], error, updates=u[0], allow_input_downcast=True)
        start_time = time.perf_counter()
        train_error = train(learn, epochs=500, verbose=True)
        print("Time taken", u[1], (time.perf_counter() - start_time))

        er_dat.plot(np.arange(len(train_error)), train_error, label=u[1])
        ax.plot(x, lstm.predict(x), label=u[1])

    er_dat.set_yscale('log', nonposy='clip')
    er_dat.set_xlabel('Iterations')
    er_dat.set_ylabel('MSE')
    sin_fig.savefig("lstm_pred.png")
    er_fig.savefig("lstm_err.png")

    er_dat.legend(loc='upper right')
    ax.legend(loc='upper right')

    sin_fig.savefig("lstm_pred.png")
    er_fig.savefig("lstm_err.png")
    plt.show()

def RNNTest():
    x = T.matrix("in")
    y = T.matrix('target')
    l1 = RecurrentLayer(1, 10, in_var=x, init_size=0.01)
    l2 = RecurrentLayer(10, 1, in_var=l1.out, init_size=0.01)
    out = l2.out

    predict = theano.function([x], out, updates=l1.updates+l2.updates)
    mse = T.mean(T.sqr(out - y))

    params = l1.params + l2.params
    (m, adadelta) = generateAdadelta(params, mse, alpha=1)

    learn = theano.function([x, y], mse, updates=adadelta)

    s_x = np.linspace(0, 10, 100)
    s_y = np.sin(s_x)

    train_error = []
    for i in range(100):
        error = learn(s_x, s_y)
        print(error)
        train_error.append(error)

    plt.plot(np.arange(len(train_error)), train_error)
    plt.show()


if __name__ == "__main__":
    LSTMTest()
