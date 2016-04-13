import numpy as np
import theano
import theano.tensor as T
from theano import config

class LSTMLayer:
    '''
        This assumes that in this recurrent net, there is a corresponding output to each input
    '''
    def __init__(self, in_size, out_size, cell_size=None, alpha=0.01, init_size=0.01,
            out_type='sigmoid', in_var=None,
            out_var=None, verbose=False):
        if cell_size == None:
            cell_size = max(in_size, out_size)
        self.alpha = alpha
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
        self.W_xf = theano.shared(value=init_size*np.random.rand(in_size, cell_size), name='x to \
                forget gate').astype(config.floatX)
        self.W_hf = theano.shared(value=init_size*np.random.rand(out_size, cell_size), name='h to \
                forget gate').astype(config.floatX)
        self.W_cf = theano.shared(value=init_size*np.random.rand(cell_size, cell_size), name='cell \
                to forget gate').astype(config.floatX)
        self.b_f = theano.shared(init_size*np.random.rand(1, cell_size), name='forget \
                bias').astype(config.floatX)

#        forget = T.nnet.sigmoid(T.dot(self.h, self.W_hf) + T.dot(self.C, self.W_cf) + T.dot(self.x, self.W_xf) + self.b_f)

        #Memories
        self.W_hm = theano.shared(value=init_size*np.random.rand(out_size, cell_size), name='h to \
                memories').astype(config.floatX)
        self.W_xm = theano.shared(value=init_size*np.random.rand(in_size, cell_size), name='x to \
                memories').astype(config.floatX)
        self.b_m = theano.shared(init_size*np.random.rand(1, cell_size), name='memory \
                bias').astype(config.floatX)

#        memories = T.tanh(T.dot(self.h, self.W_hm) + T.dot(self.x, self.W_xm) + self.b_m)

        #Remember Gate
        self.W_hr = theano.shared(value=init_size*np.random.rand(out_size, cell_size), name='h to \
                remember').astype(config.floatX)
        self.W_cr = theano.shared(value=init_size*np.random.rand(cell_size, cell_size), name='cell to \
        remember').astype(config.floatX)
        self.W_xr = theano.shared(value=init_size*np.random.rand(in_size, cell_size), name='x to \
                remember').astype(config.floatX)
        self.b_r = theano.shared(value=init_size*np.random.rand(1, cell_size), name='remember \
                bias').astype(config.floatX)

#        remember = T.nnet.sigmoid(T.dot(self.h, self.W_hr) + T.dot(self.C, self.W_cr) + T.dot(self.x, self.W_xr) + self.b_r)

        #Output
        self.W_co = theano.shared(value=init_size*np.random.rand(cell_size, out_size), 
            name='cell to out').astype(config.floatX)
        self.W_ho = theano.shared(value=init_size*np.random.rand(out_size, out_size),
            name='hidden to out').astype(config.floatX)
        self.W_xo = theano.shared(value=init_size*np.random.rand(in_size, out_size), name='x to out')
        self.b_o = theano.shared(value=init_size*np.random.rand(1, out_size), name='out bias')

        #Hidden
        self.W_ch = theano.shared(value=init_size*np.random.rand(cell_size, out_size), name='cell to hidden')
        self.W_hh = theano.shared(value=init_size*np.random.rand(out_size, out_size), 
            name='hidden to hidden')
        self.W_xh =theano.shared(value=init_size*np.random.rand(in_size, out_size), name='x to hidden')
        self.b_h =theano.shared(value=init_size*np.random.rand(1, out_size), name='bias for hidden')


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

class LSTM():

    def __init__(self, *dim, **kwargs):
        self.alpha = kwargs.get('alpha', 0.01)
        self.momentum = kwargs.get('momentum', 0)
        rprop = kwargs.get('rprop', False)
        out_type = kwargs.get('out_type', 'sigmoid')
        self.layers = []
        verbose = kwargs.get('verbose', False)
        init_size = kwargs.get('init_size', 0.01)
        x = T.matrix('Input')
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
