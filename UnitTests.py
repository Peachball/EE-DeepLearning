import unittest
import theano
import theano.tensor as T
import numpy as np

class TestTrainers(unittest.TestCase):

    def setUp(self):
        pass

    def test_miniRecurrentLearning(self):
        from RecurrentNetworks import miniRecurrentLearning
        from RecurrentNetworks import LSTM
        from DeepLearning import generateRmsProp

        lstm = LSTM(1, 5, 1)
        print("lstm set up ... Compiling functions now")
        lstm.reset()
        predict = theano.function([lstm.x], lstm.out)

        error = T.mean(T.sqr(lstm.out- lstm.y))

        print("Finding gradients")
        (data, rmsupdates) = generateRmsProp(lstm.params, error, alpha=0.01)

        learn = theano.function([lstm.x, lstm.y], error, updates=rmsupdates)

        print("Functions set up, preparing data")

        x_data = np.arange(2000).reshape(2000, 1)
        y_data = np.sin(x_data)


        miniRecurrentLearning(x_data, y_data, 10, learn, predict, verbose=True,
                epochs=100, miniepochs=1)

        assert np.polyfit(np.arange(len(train_error)), train_error, 1) < 0
