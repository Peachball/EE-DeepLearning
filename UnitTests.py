import unittest
import theano
import theano.tensor as T
import numpy as np

class TestTrainers(unittest.TestCase):

    def setUp(self):
        pass

    # def test_miniRecurrentLearning(self):
        # from RecurrentNetworks import miniRecurrentLearning
        # from RecurrentNetworks import LSTM
        # from DeepLearning import generateRmsProp

        # lstm = LSTM(1, 5, 1)
        # print("lstm set up ... Compiling functions now")
        # lstm.reset()
        # predict = theano.function([lstm.x], lstm.out)

        # error = T.mean(T.sqr(lstm.out- lstm.y))

        # print("Finding gradients")
        # (data, rmsupdates) = generateRmsProp(lstm.params, error, alpha=0.01)

        # learn = theano.function([lstm.x, lstm.y], error, updates=rmsupdates)

        # print("Functions set up, preparing data")

        # x_data = np.arange(2000).reshape(2000, 1)
        # y_data = np.sin(x_data)


        # miniRecurrentLearning(x_data, y_data, 10, learn, predict, verbose=True,
                # epochs=100, miniepochs=1)

        # assert np.polyfit(np.arange(len(train_error)), train_error, 1) < 0

    def test_FT_to_wav(self):
        from MusicLoader import FT_to_wav
        from MusicLoader import wav_to_FT
        X_dat = 10 * np.sin(np.arange(1000).reshape(500, 2))

        XL_dat = 10 * np.sin(np.arange(10000).reshape(5000, 2))

        assert np.sum(np.abs(X_dat[:256] - FT_to_wav(wav_to_FT(X_dat)))) < 1
        assert np.sum(np.abs(XL_dat[:4864] - FT_to_wav(wav_to_FT(XL_dat)))) < 1


class EyeConvnetTest(unittest.TestCase):
    def setup_dataTest(self):
        pass

def test_LSTM_constructor():
    from RecurrentNetworks import LSTM
    try:
        l =  LSTM(4, 4, 4, verbose=True)
    except:
        assert False

def test_normalize():
    from DeepLearning import normalize, scaleBack
    x = np.random.uniform(low=-100, high=100, size=(100,
        100)).astype('float32')

    scale, new_x = normalize(x)

    print(new_x.max(), new_x.min())
    print(np.sum(scaleBack(new_x, scale) - x))
    assert np.sum(scaleBack(new_x, scale) - x) < 0.001

if __name__ == "__main__":
    test_normalize()
