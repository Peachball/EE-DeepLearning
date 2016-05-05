from DeepLearning import generateRpropUpdates
import pickle
import theano
from DeepLearning import miniBatchLearning
from DeepLearning import *
from RecurrentNetworks import LSTM
from RecurrentNetworks import miniRecurrentLearning
import numpy as np
import scipy.io.wavfile as wavUtil

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

def testLSTM():
    data = convertMusicFile(0)
    lstm = LSTM(1000, 1000, verbose=True, init_size=0.1, out_type='linear')


    x = data[:-1]
    '''
    print("Writing music file...")
    generateMusicFile(x, 'test.wav')
    print("Done!")
    '''
    y = data[1:]
    print(x.shape)

    (rprop, rupdates) = generateRpropUpdates(lstm.params, lstm.error, 
            init_size=0.1, verbose=True)
    (adamstorage, adam) = generateAdam(lstm.params, lstm.error, alpha=1,
            verbose=True)

    learnFunc = theano.function([lstm.x, lstm.y], lstm.error, updates=adam)
    lstm.reset()
    train_error = miniRecurrentLearning(x, y, 10, learnFunc, lstm.predict, 
            verbose=True, miniepochs=10)
    plt.plot(np.arange(len(train_error)), train_error)
    plt.show()


if __name__ == '__main__':
    testLSTM()
