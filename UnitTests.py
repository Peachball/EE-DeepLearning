import unittest
import theano
import theano.tensor as T

class TestTrainers(unittest.TestCase):

    def setUp(self):
        pass

    def test_miniRecurrentLearning(self):
        from RecurrentNetworks import miniRecurrentLearning
        from RecurrentNetworks import LSTM
        from DeepLearning import generateRmsProp

        lstm = LSTM(1, 5, 1)
        lstm.reset()
        predict = theano.function([lstm.x], lstm.prediction)

        error = T.mean(T.sqr(lstm.prediction - lstm.y))

        (data, rmsupdates) = generateRmsProp(lstm.params, error, alpha=0.01)

        learn = theano.function([lstm.x, lstm.y], error, updates=rmsupdates)

        x_data = np.arange(2000)
        y_data = np.sin(x_data)

        miniRecurrentLearning(x_data, y_data, 10)
