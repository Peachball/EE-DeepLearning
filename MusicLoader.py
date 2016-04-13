from DeepLearning import generateRpropUpdates
import pickle
import theano
from DeepLearning import miniBatchLearning
from RecurrentNetworks import LSTM
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
    data.resize(data.shape[0]//100, inputsize)
    return data

def testLSTM():
    data = convertMusicFile(0)
    lstm = LSTM(1000, 1000, verbose=True, init_size=0.1, out_type='linear')
    
    (rprop, rupdates) = generateRpropUpdates(lstm.params, lstm.error, init_size=0.1, verbose=True)

    learnFunc = theano.function([lstm.x, lstm.y], lstm.error, updates=rupdates)

    x = data[:-1]
    y = data[1:]
    savefile = open("lstm_test.pkl", "wb")
    savefile.close()
    miniBatchLearning(x, y, -1, learnFunc, verbose=True, epochs=10)
    

if __name__ == '__main__':
    testLSTM()
