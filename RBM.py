import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt

class RBMLayer:
    def __init__(self in_size, out_size, visible_type='binary', hidden_type='binary', nonlinearity=T.nnet.sigmoid):
        self.in_size = in_size
        self.out_size = out_size
        