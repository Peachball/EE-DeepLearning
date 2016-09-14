import numpy as np
import theano
import theano.tensor as T
from theano import config
from DeepLearning import *
import matplotlib.pyplot as plt
import time
import math

class RecurrentLayer:

    def __init__(self, in_size, out_size, in_var=T.matrix('input'),
            init_size=0.01, verbose=False, hidden_size=None,
            nonlinearity=T.nnet.sigmoid, h_nonlinearity=T.nnet.sigmoid):
        x = in_var
        self.x = x
        if hidden_size == None:
            hidden_size = max(in_size, out_size)

        w_io = init_weights((in_size, out_size), init_type='xavier',
                scale=init_size)

        w_ih = init_weights((in_size, hidden_size),
        init_type='xavier', scale=init_size)

        w_hh = init_weights((hidden_size, hidden_size),
                init_type='xavier', scale=init_size)
        b_h = init_weights((hidden_size), init_type='xavier',
                scale=init_size)

        w_ho = init_weights((hidden_size, out_size),
                init_type='xavier', scale=init_size)
        b_o = init_weights((out_size), init_type='xavier',
                scale=init_size)

        self.hidden = theano.shared(
                value=np.zeros((hidden_size)).astype(theano.config.floatX))

        def recurrence(x, h_tm1):
            h_t =  h_nonlinearity(T.dot(x, w_ih) + T.dot(h_tm1, w_hh) +
                    b_h)
            out = nonlinearity(T.dot(x, w_io) + T.dot(h_tm1, w_ho) +
                    b_o)
            return [out, h_t]

        ([out, hidden], updates) = theano.scan(recurrence,
                sequences=x, outputs_info=[None, self.hidden], n_steps=x.shape[0])

        self.out = out
        self.updates = [(self.hidden, hidden[-1])]

        self.params = [w_ih, w_hh, b_h, w_io, w_ho, b_o]

    def reset(self):
        shape = self.hidden.get_value().shape
        self.hidden.set_value(np.zeros(shape).astype(theano.config.floatX))

class RNN:

    def __init__(self, *dim, **kwargs):
        init_size = kwargs.get('init_size', 0.01)
        verbose = kwargs.get('verbose', False)

        x = kwargs.get('in_var', T.matrix("input"))
        layers = []
        layers.append(RecurrentLayer(dim[0], dim[1], init_size = init_size, in_var=x))
        out_type = kwargs.get('out_type', 'sigmoid')

        nonlinearity = T.nnet.sigmoid
        for i in range(1, len(dim) -1):
            if i == len(dim) - 2:
                if out_type == 'linear':
                    nonlinearity = lambda x: x
                if out_type == 'sigmoid':
                    nonlinearity = T.nnet.sigmoid
                if out_type == 'tanh':
                    nonlinearity = T.tanh
            layers.append(RecurrentLayer(dim[i], dim[i+1], init_size=init_size,
                in_var=layers[-1].out, nonlinearity=nonlinearity))

        self.out = layers[-1].out
        updates = []
        self.params = []
        for l in layers:
            updates = updates + l.updates
            self.params = self.params + l.params
        self.updates = updates
        self.layers = layers

    def reset(self):
        for l in self.layers:
            l.reset()

class CWLayer:
    def __init__(self, input_size, output_size, hidden_units, modules,
            in_var=T.matrix(), init_type='xavier',
            init_size=-1, nonlinearity=T.tanh, h_nonlinearity=T.tanh,
            periods=None):
        x = in_var
        params = []

        if periods is None:
            val = np.arange(modules).astype(theano.config.floatX)
            val = 2 ** val
            p = theano.shared(val)
        else:
            p = theano.shared(np.array(periods))

        self.x = x
        time = theano.shared(np.array(0).astype(theano.config.floatX))
        self.time = time

        self.hidden = theano.shared(np.zeros(hidden_units)
                .astype(theano.config.floatX))
        self.modules = modules

        weight_val = init_weights((hidden_units, hidden_units), scale=0.05,
                shared_var=False)
        sizeof_mod = math.ceil(hidden_units / modules)

        if modules <= 1 or modules > hidden_units:
            raise Exception("Invalid Module size")

        for i in range(1, modules):
            weight_val[(i * sizeof_mod):,:(i * sizeof_mod)] = 0
        self.wh = theano.shared(weight_val.astype(theano.config.floatX))
        self.bh = init_weights(hidden_units)
        self.wi = init_weights((input_size, hidden_units), init_type=init_type,
                scale=init_size)

        self.wo = init_weights((hidden_units, output_size), init_type=init_type,
                scale=init_size)
        self.bo = init_weights(output_size, scale=init_size)

        def recurrence(x, cur_time, prev_hidden):
            act_modules = modules - T.argmin((cur_time % p)[::-1])

            update_indices = act_modules * sizeof_mod

            w_subtensor = self.wh[:update_indices]
            b_subtensor = self.bh[:update_indices]

            inp_updates = T.dot(x, self.wi)[:update_indices]
            pre_new_h = T.dot(w_subtensor, prev_hidden) + inp_updates + b_subtensor
            new_h = T.set_subtensor(prev_hidden[:update_indices],
                    h_nonlinearity(pre_new_h))

            out = nonlinearity(T.dot(new_h, self.wo) + self.bo)

            return (cur_time+1.0), new_h, out

        ([times, hiddens, outputs], updates) = theano.scan(recurrence,
                                            sequences=[x],
                                            outputs_info=[self.time,
                                                self.hidden, None],
                                            n_steps=x.shape[0])
        self.out = outputs
        updates.update({self.hidden: hiddens[-1]})
        updates.update({self.time: times[-1]})
        self.updates = [(self.hidden, hiddens[-1]), (self.time, times[-1])]

        self.params = [self.wh, self.bh, self.wi, self.wo, self.bo]

    def reset(self):
        shape = self.hidden.get_value().shape
        self.hidden.set_value(np.zeros(shape).astype(theano.config.floatX))
        self.time.set_value(0.0)

class GRULayer:
    def __init__(self, in_size, out_size, cell_size=None, init_size=-1,
            nonlinearity=T.tanh, in_var=T.matrix(), init_type='uniform'):
        if not cell_size:
            cell_size = max(in_size, out_size)
        x = in_var
        self.x = x

        b_z = init_weights(cell_size, scale=init_size, init_type=init_type)
        b_r = init_weights(cell_size, scale=init_size, init_type=init_type)
        b_h = init_weights(cell_size, scale=init_size, init_type=init_type)
        b_o = init_weights(out_size, scale=init_size, init_type=init_type)

        W = init_weights((in_size, 3 * cell_size + out_size), scale=init_size,
                init_type=init_type)
        U = init_weights((cell_size, 2 * cell_size), scale=init_size,
                init_type=init_type)

        U_r = init_weights((cell_size, cell_size), scale=init_size,
                init_type=init_type)
        U_o = init_weights((cell_size, out_size), scale=init_size,
                init_type=init_type)

        self.params = [b_z, b_r, b_h, b_o, W, U, U_r, U_o]

        def recurrence(x, h_tm1):
            inp_result = T.dot(x, W)
            hid_result = T.dot(h_tm1, U)
            r_t = T.nnet.sigmoid(inp_result[:cell_size] +
                    hid_result[:cell_size] + b_r)
            ht_t = T.tanh(inp_result[(cell_size):(2*cell_size)] +
                    T.dot((h_tm1*r_t), U_r) + b_h)
            z_t = T.nnet.sigmoid(inp_result[(2*cell_size):(3*cell_size)] +
                    hid_result[cell_size:(2*cell_size)] + b_z)

            h_t = ((1.0 - z_t) * h_tm1) + (z_t * ht_t)
            output = nonlinearity(
                    inp_result[(3*cell_size):(3*cell_size+out_size)] +
                    T.dot(h_t, U_o) + b_o)


            return h_t, output

        self.hidden = theano.shared(
                np.zeros(cell_size).astype(theano.config.floatX))
        ([hiddens, output], updates) = theano.scan(recurrence,
                                        sequences=[x],
                                        outputs_info=[self.hidden, None],
                                        n_steps=x.shape[0])

        self.new_h = hiddens[-1]


        self.out = output
        updates.update({self.hidden: hiddens[-1]})
        self.updates = [(self.hidden, hiddens[-1])]

    def reset(self):
        shape = self.hidden.get_value().shape
        self.hidden.set_value(np.zeros(shape).astype(theano.config.floatX))

class GRU:
    def __init__(self, *dim, **kwargs):
        self.x = kwargs.get('in_var', T.matrix())
        h_nonlinearity = kwargs.get('h_nonlinearity', T.tanh)
        nonlinearity = kwargs.get('nonlinearity', T.nnet.sigmoid)
        init_size = kwargs.get('init_size', -1)
        verbose = kwargs.get('verbose', False)

        layers = []
        nl = h_nonlinearity
        if len(dim) <= 2:
            nl = nonlinearity
        l = GRULayer(dim[0], dim[1], in_var=self.x, nonlinearity=nl,
                init_size=init_size)
        layers.append(l)

        for i in range(1, len(dim)-1):
            if i == len(dim) - 2:
                nl = nonlinearity
            l = GRULayer(dim[i], dim[i+1], in_var=layers[-1].out,
                    nonlinearity=nl, init_size=init_size)
            layers.append(l)

        self.params = []
        self.updates = []
        for l in layers:
            self.params += l.params
            self.updates += l.updates

        self.out = layers[-1].out
        self.predict = theano.function([self.x], self.out, updates=self.updates,
                allow_input_downcast=True)
        self.layers = layers
        print("Done constructing gru")
        return

    def reset(self):
        for l in self.layers:
            l.reset()
        return

class LSTMLayer:
    '''
        This assumes that in this recurrent net, there is a corresponding output to each input
    '''
    def __init__(self, in_size, out_size, cell_size=None, init_size=-1,
            out_type='sigmoid', in_var=None, verbose=False,
            mem_bias=1, initialization_type='uniform'):
        if cell_size is None:
            cell_size = max(in_size, out_size)
        self.in_size = in_size
        self.out_size = out_size
        self.cell_size = cell_size
        self.C = theano.shared(value=np.zeros((cell_size,))
                .astype(theano.config.floatX), name='Longterm Cell')
        self.h = theano.shared(value=np.zeros((cell_size,))
                .astype(theano.config.floatX), name='Shortterm Cell')
        if in_var == None:
            x = T.matrix(name='Input Matrix')
        else:
            x = in_var
        if verbose:
            print('Constants have been initalized')

        self.x = x

        self.W = init_weights((in_size, cell_size * 4),
                init_type=initialization_type, scale=init_size,
                name='x to cell weights')
        self.U = init_weights((cell_size, cell_size * 2),
                init_type=initialization_type, scale=init_size,
                name='cell to gate weights')
        self.W_h = init_weights((cell_size, cell_size * 4),
                init_type=initialization_type, scale=init_size,
                name='hidden to cell weights')

        # Order of rows:
        #   Input Gate, Forget Gate, Memory Generation, Hidden Generation
        #   Output Generation (not based on hidden)


        #Forget gate
        self.b_f = init_weights(cell_size,
                init_type='bias', scale=mem_bias,
                name='forget gate bias')

        #Memories
        self.b_m = init_weights(cell_size,
                init_type=initialization_type, scale=init_size,
                name='memory bias')

        #Remember Gate
        self.b_r = init_weights(cell_size, init_type=initialization_type,
                scale=init_size, name='rem gate bias')

        #Output
        self.W_ho = init_weights((cell_size, out_size),
                init_type=initialization_type, scale=init_size,
                name='hidden to out weight')
        self.W_co = init_weights((cell_size, out_size),
                init_type=initialization_type, scale=init_size,
                name='cell to out weight')
        self.W_xo = init_weights((in_size, out_size),
                init_type=initialization_type, scale=init_size,
                name='inp to out weight')
        self.b_o = init_weights(out_size, init_type=initialization_type,
                scale=init_size, name='out bias')

        #Hidden
        self.b_h = init_weights(cell_size, init_type=initialization_type,
                scale=init_size, name='hidden bias')

        self.params = [self.W, self.U, self.W_h, self.b_f, self.b_m, self.b_r,
                self.W_ho, self.W_co, self.W_xo, self.b_o, self.b_h]
        if verbose:
            print('Weights have been initalized')

        updates, output = self.getOutput(x)
        if verbose:
            print('Recurrence has been set up')

        self.out = output
        if out_type == 'sigmoid':
            self.out = T.clip(T.nnet.sigmoid(output), 0.001, 0.999)
        if out_type == 'tanh':
            self.out = T.clip(T.tanh(output), -0.999, 0.999)
        self.updates = updates

    def getOutput(self, x):
        cell_size = self.cell_size
        def recurrence(x, h_tm1, c_tm1):
            x_to_cell = T.dot(x, self.W)
            cell_to_gate = T.dot(c_tm1, self.U)
            hid_to_cell = T.dot(h_tm1, self.W_h)

            rem = T.nnet.sigmoid(x_to_cell[:cell_size]
                    + cell_to_gate[:cell_size]
                    + hid_to_cell[:cell_size]
                    + self.b_r)
            rem.name = 'Memory Gate'
            forget = T.nnet.sigmoid(x_to_cell[cell_size:(cell_size*2)]
                    + cell_to_gate[cell_size:(cell_size*2)]
                    + hid_to_cell[cell_size:(cell_size*2)]
                    + self.b_f)
            forget.name = 'Forget Gate'
            mem = T.tanh(x_to_cell[(cell_size*2):(cell_size*3)]
                     + hid_to_cell[(cell_size*2):(cell_size*3)]
                    + self.b_m)
            mem.name = 'Memories'

            h_t = T.tanh(mem) * T.nnet.sigmoid(x_to_cell[(cell_size*3):]
                                              + hid_to_cell[(cell_size*3):]
                                              + self.b_h)

            c_t = c_tm1 * forget + rem * mem

            z = T.dot(c_t, self.W_co) + T.dot(h_tm1, self.W_ho) \
                    + T.dot(x, self.W_xo) + self.b_o
            return [h_t, c_t, z]

        ([hidden, cell_state, output], _) = theano.scan(fn=recurrence,
                sequences=[x],
                outputs_info=[self.h, self.C, None],
                n_steps=x.shape[0])
        updates = [(self.h, hidden[-1]), (self.C, cell_state[-1])]
        return updates, output

    def reset(self):
        self.C.set_value(np.zeros(self.C.shape.eval())
                .astype(theano.config.floatX))
        self.h.set_value(np.zeros(self.h.shape.eval())
                .astype(theano.config.floatX))

class LSTM():

    def __init__(self, *dim, **kwargs):
        """
        Initialize a multi layer lstm only classifier

        args:
            dim - (tuple of ints)
                Layer sizes of model (parallels neural network)
            out_type - string
                Nonlinearity that the lstm will apply on output units
            verbose - bool
                If true, output information about the construction of model
            init_size - float(0, inf)
                Define the range of values that the paramaters of the model can
                have
            in_var - theano.tensor.matrix()
                Define the input (from perhaps another model)
        """
        out_type = kwargs.get('out_type', 'sigmoid')
        self.layers = []
        verbose = kwargs.get('verbose', False)
        init_size = kwargs.get('init_size', 0.001)
        x = kwargs.get("in_var", T.matrix('Input'))
        y = kwargs.get("out_var", T.matrix('Output'))
        self.x = x
        self.y = y

        init_size = kwargs.get('init_size', -1)
        self.layers.append(LSTMLayer(dim[0], dim[1], in_var=x, verbose=False))
        layer_out_type = 'tanh'
        for i in range(1, len(dim) - 1):
            if i == len(dim) - 2:
                layer_out_type = out_type
            self.layers.append(LSTMLayer(dim[i], dim[i+1],
                in_var=self.layers[-1].out,
                init_size=init_size,
                out_type=layer_out_type))


        if verbose:
            print('Number of layers:' + str(len(self.layers)))
            print('Finished with initialization of layers -> Defining prediction')

        #Defining updates for all layers:
        layerUpdates = []
        for l in self.layers:
            layerUpdates += l.updates

        #Define prediction:
        prediction = self.layers[-1].out
        self.predict = theano.function([x], prediction, updates=layerUpdates
                ,allow_input_downcast=True)
        self.out = prediction
        self.updates = layerUpdates

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


def miniRecurrentLearning(x, y, batchSize, learn, predict, reset, verbose=False,
        epochs=1, miniepochs=1, save=None, saveiters=None, strides=1, f=None):
    """
    Train model on parts of a time series at a time
    e.g. given time seires : 1, 2, 3, 4, 5, 6
        Train 1 and 2, then 2 and 3, etc

    Variable intepretations:
        x - numpy array : each row is a data sample
                          each column is a feature
        y - numpy array : same as x, but for labels

        batchSize - int(0, inf) : size of the time series segment
                                  The example would have a batchSize of 2

        learn - void func(data, labels) : function that trains the model
        predict - labels func(data) : function that updates the internal state of
                                  the model
        verbose - boolean : display train error to stdout or not
        epochs - int(0, inf) : number of times to iterate through train data
        miniepochs - int(0, inf) : number of times to train one time series
                                   segment
                                   In the example, that would be the number of
                                   times to train on "1 and 2", then "2 and 3",
                                   etc.
    """

    train_error = []
    if batchSize <= 0: batchSize = x.shape[0]
    iterations = 0
    for j in range(epochs):
        for batch in range(0, x.shape[0], strides):
            train_error = train_error + miniBatchLearning(x[batch:batch+batchSize],
                    y[batch:batch+batchSize], -1, learn, verbose=False,
                    epochs=miniepochs)
            iterations += 1
            if verbose: print("Epoch: ", "%.4f" % ((j) + batch/x.shape[0]),
                    "Error: ", (train_error[-1]))
            if save and saveiters:
                if iterations % saveiters == 0:
                    save()
            if not f is None:
                f.write(str(train_error[-1]) + '\n')
                f.flush()
            predict(x[batch:batch+strides])
        reset()
    return train_error

def gen_sine_data(amount=100):
    x = np.linspace(0, 10, amount).reshape(-1, 1).astype(theano.config.floatX)
    y = np.sin(x)
    return x, y

def LSTMLayerTest():
    x_dat, y = gen_sine_data()
    x = T.matrix()
    updates = []
    params = []
    m = LSTMLayer(1, 1, cell_size=7, in_var=x)
    updates = updates + m.updates
    params += m.params
    out = m.out

    # m = LSTMLayer(10, 1, cell_size=7, in_var=m.out)
    # updates = updates + m.updates


    predict = theano.function([x], m.out, updates=updates,
            allow_input_downcast=True)
    y_ = T.matrix()
    error = T.mean(T.sqr(out - y_))
    print(predict(x_dat[:23]).shape)
    (storage, grad_updates) = generateRmsProp(params, error, alpha=0.01,
            verbose=True)
    learn = theano.function([x, y_], error, updates=grad_updates,
            allow_input_downcast=True)


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
    (m, adam) = generateAdam(params, mse, alpha=0.01)
    (r, rprop) = generateRpropUpdates(params, mse, init_size=0.01)

    learn = theano.function([x, y], mse, updates=rprop)

    print("Compiled Learn Function!")

    s_x = np.linspace(0, 10, 100).reshape(100, 1)
    s_y = np.sin(s_x)

    train_error = []
    for i in range(1000):
        error = learn(s_x, s_y)
        print(error)
        train_error.append(error)

    plt.plot(np.arange(len(train_error)), train_error)
    plt.show()

def CWRNNTest():
    x, y = gen_sine_data()
    model = CWLayer(1, 1, 16, 4)

    correct = T.matrix()
    out = model.out

    error = T.mean(T.sum(T.sqr(out - correct), axis=1))

    (storage, grad_updates) = generateRmsProp(model.params, error, alpha=0.01)

    learn = theano.function([model.x, correct], error, updates=grad_updates,
            allow_input_downcast=True)
    predict = theano.function([model.x], out, updates=model.updates,
            allow_input_downcast=True)

    for i in range(1000):
        print(learn(np.zeros(y.shape), y))

    plt.plot(x, predict(np.zeros(y.shape)))
    plt.show()

    return

def GRUTest():
    x, y = gen_sine_data()
    model = GRULayer(1, 1, 5, nonlinearity=lambda x:x)

    predict = theano.function([model.x], model.out, updates=model.updates,
            allow_input_downcast=True)

    y_ = T.matrix()
    error = T.mean(T.sum(T.sqr(model.out - y_), axis=1))

    (storage, gradUpdates) = generateRmsProp(model.params, error, alpha=0.01)

    updates = model.updates + gradUpdates

    learn = theano.function([model.x, y_], error, updates=gradUpdates)

    for i in range(10):
        print(model.hidden.get_value())
        (predict(np.zeros((100, 1))))
        print(model.hidden.get_value())


    plt.plot(x, predict(x))
    plt.show()

if __name__ == "__main__":
    LSTMTest()
