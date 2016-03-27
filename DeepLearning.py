from __future__ import print_function
import theano.tensor as T
import theano
import numpy as np

class Layer:
    def __init__(self, in_size, out_size, no_compile=True, layer_type='sigmoid', in_var=None):
        if not layer_type in ['sigmoid', 'tanh', 'lstm', 'rnn']:
            raise Exception('Layer type is invalid: ' + str(layer_type))
        
        if in_var==None:
            x = T.matrix('Input')
        else:
            x = in_var

        if layer_type in ['sigmoid', 'tanh']:
            pass


    def predict(self, x):
        pass

    def getParams(self):
        pass

    def getOutput(self):
        pass
