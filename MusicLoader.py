from DeepLearning import generateRpropUpdates
import pickle
import theano
from DeepLearning import miniBatchLearning
from DeepLearning import *
from RBM import *
from RecurrentNetworks import LSTM
from RecurrentNetworks import miniRecurrentLearning
import numpy as np
import scipy.io.wavfile as wavUtil
import matplotlib.pyplot as plt
from RecurrentNetworks import *

def convertMusicFile(index, inputsize=1000):
    filename = 'musicDataSet/' + str(index) + '.wav'
    samplerate, data = wavUtil.read(filename)

    index1 = 0
    index2 = -1
    while data[index1][0] == 0 and data[index1][1] == 0:
        index1 += 1

    while data[index2][0] == 0 and data[index2][1] == 0:
        index2 -= 1

    data = data[index1:(index2 + 1)]
    data = data.flatten()
    data.resize(data.shape[0]//inputsize, inputsize)
    return data

def generateMusicFile(arr, f):
    data = arr.flatten().reshape(arr.size/2, 2)
    wavUtil.write(f, 44100, data)

def testPrediction(predict):
    data = convertMusicFile(0)[:1000]
    newSong = generateMusicFile(predict(data), open("test.wav", 'wb'))

def testLSTM():
    data = convertMusicFile(0)
    lstm = LSTM(1000, 1000, 1000, verbose=True, init_size=0.1, out_type='linear')


    x = data[:-1]
    '''
    print("Writing music file...")
    generateMusicFile(x, 'test.wav')
    print("Done!")
    '''
    y = data[1:]
    generateMusicFile(x[:1000], open('test.wav', 'wb'))
    print(x.shape)

    (rprop, rupdates) = generateRpropUpdates(lstm.params, lstm.error, 
            init_size=0.1, verbose=True)
    (adamstorage, adam) = generateAdam(lstm.params, lstm.error, alpha=1,
            verbose=True)

    learnFunc = theano.function([lstm.x, lstm.y], lstm.error, updates=adam)
    lstm.reset()

    train_error = miniBatchLearning(x[:1000], y[:1000], -1, learnFunc,
            verbose=True, epochs=100)
    '''
    train_error = miniRecurrentLearning(x, y, 10, learnFunc, lstm.predict, 
            verbose=True, miniepochs=10)
    '''
    plt.plot(np.arange(len(train_error)), train_error)
    plt.show()

def testRNN():
    x = T.matrix('input')
    y = T.matrix('output')
    rnn = RecurrentLayer(1000, 1000, init_size=0.1, in_var=x, 
            nonlinearity=lambda x: x)

    data = convertMusicFile(0, inputsize=1000)
    err = T.mean(T.sqr(rnn.out - y))

            verbose=False)

    (stor, adam) = generateAdam(ae.params, mse, alpha=0.001)
    (stora, rms) = generateRmsProp(ae.params, mse, alpha=0.01)
    (storag, momentum) = generateMomentumUpdates(ae.params, mse, alpha=1e-9,
            momentum=0)

    learn = theano.function([ae.x, y], mse, updates=rupdates)

    train_error = miniBatchLearning(data[:1000], data[:1000], -1, learn, verbose=True,
            epochs=1000)

    plt.plot(np.arange(len(train_error)), train_error)
    plt.yscale('log')
    plt.show()
    predict = theano.function([ae.x], ae.reconstructed * scaleFactor)
    testPrediction(predict)

def testRBM():
    data = convertMusicFile(0)
    rbm = RBMLayer(1000, 900, persistent_updatesize=500)
    scaleFactor = data.max()
    data = data / data.max()

    y = T.matrix()

    cdupdates = rbm.CDUpdates(rbm.in_var, alpha=0.0001)

    mse = T.mean(T.sqr((rbm.out - y) * scaleFactor))

    learn = theano.function([rbm.in_var, y], mse, updates=cdupdates)

    train_error = rbm.miniBatch(learn, data, verbose=True, epochs=10)

    plt.plot(np.arange(len(train_error)), train_error)
    plt.show()


if __name__ == '__main__':
    testRNN()
