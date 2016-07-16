import pickle
import theano
from DeepLearning import *
from RBM import *
from RecurrentNetworks import LSTM
from RecurrentNetworks import miniRecurrentLearning
import numpy as np
import scipy.io.wavfile as wavUtil
import matplotlib.pyplot as plt
from RecurrentNetworks import *
import scipy.fftpack
import warnings

SAMPLE_RATE = 44100

def convertMusicFile(index, inputsize=1000):
    filename = 'musicDataSet/' + str(index) + '.wav'
    samplerate, data = wavUtil.read(filename)

    #Get rid of quiet parts in the beginning and end
    index1 = 0
    index2 = -1
    while data[index1][0] == 0 and data[index1][1] == 0:
        index1 += 1
    while data[index2][0] == 0 and data[index2][1] == 0:
        index2 -= 1
    data = data[index1:(index2 + 1)]

    return data

def generateMusicFile(arr, f):
    data = arr.reshape(arr.size/2, 2).astype('int16')
    wavUtil.write(f, 44100, data)

def testPrediction(predict):
    data = convertMusicFile(0)[:1000]
    newSong = generateMusicFile(predict(data), open("test.wav", 'wb'))

def wav_to_FT(data, chunk_size=1024):
    """
        Convert list of numbers representing sound file into it's respective
        fourier transform

        Variable interp.
            data       - 1st dimension represents time
                       - 2nd dimension represents audio channels
            chunk_size - number of columns of output array (number of features)

    NOTE: The chunk_size here is different than the chunk_size variable used
          for the inverse method (wav_to_FT())
    """

    channels = data.shape[1]
    if chunk_size % (channels * 2) != 0:
        print("Warning: output data will not have the desired number of"
              "columns")

    output = np.zeros((data.size * 2 / chunk_size, chunk_size))

    row_counter = 0
    for i in range(0, data.shape[0], chunk_size // (2 * channels)):
        row = []
        for j in range(channels):
            comp = np.fft.fft(data[i:i + (chunk_size / (2 * channels)), j])
            row = row + list(comp.real) + list(comp.imag)
        if len(row) != chunk_size:
            break
        output[row_counter] = row
        row_counter += 1

    return output

def FT_to_wav(data, channels=2):
    """
        Convert result of fourier transform into a wav file

        Paramaters:
            data     - 2d array output of wav_to_FT()
            channels - number of channels that the original sound file had
    """
    output = np.zeros((data.size / (2 * channels), channels))

    time = 0

    chunk_time_size = data.shape[1] / (2 * channels)
    for row in data:
        time_length = data.shape[1] // channels

        curchannel = 0
        for c in range(0, data.shape[1], time_length):
            ft = row[c:(c+time_length/2)] + 1j * \
                row[(c+time_length/2):(c+time_length)]
            org = np.fft.ifft(ft)
            output[time:(time+chunk_time_size),curchannel] = org
            curchannel += 1
        time += chunk_time_size
    return output

def viewFT(x):
    x = x.reshape(x.size/2, 2)
    xf1 = scipy.fftpack.fft(x)
    scale = np.linspace(0, 1/SAMPLE_RATE, x.shape[0])
    plt.plot(scale, xf1)
    plt.show()

def testLSTM():
    theano.config.floatX = 'float32'
    orgdata = convertMusicFile(0)
    scale, data = normalize(wav_to_FT(orgdata))
    lstm = LSTM(1024, 1024, verbose=True, init_size=0.1, out_type='linear')


    x = data[:-1]
    # y = data[1:]
    y = x

    # (rprop, rupdates) = generateRpropUpdates(lstm.params, lstm.error,
            # init_size=0.1, verbose=True)
    # (adamstorage, adam) = generateAdam(lstm.params, lstm.error, alpha=0.01,
            # verbose=True)
    (storage , rms) = generateRmsProp(lstm.params, lstm.error, alpha=0.001,
            verbose=True)


    # train_error = miniBatchLearning(x[:1000], y[:1000], -1, learnFunc,
            # verbose=True, epochs=100)

    savefile = "musicloader"
    params = lstm.params + storage
    def test():
        print("Generating sample music file")
        result = FT_to_wav(scaleBack(data, scale))
        # print(np.sum(np.abs(
            # orgdata[:result.shape[0]] - FT_to_wav(scaleBack(data, scale)))))
        generateMusicFile(
                FT_to_wav(scaleBack(lstm.predict(x[:1000]), scale)).real
                , open('test' + str(test.count) + '.wav', 'wb'))
        test.count += 1
        print("Done")

    test.count = 0

    def save():
        saveParams(params, savefile)
        test()

    def load():
        print("Attempting to load paramaters")
        loadParams(params, savefile + ".npz")
        print("Successfully loaded")

    test()
    try:
        load()
    except:
        print("Failed to load params")
        save()

    learnFunc = theano.function([lstm.x, lstm.y], lstm.error, updates=rms
            ,allow_input_downcast=True)
    lstm.reset()
    train_error = miniRecurrentLearning(x, y, 10, learnFunc, lstm.predict,
            verbose=True, miniepochs=1, save=save, saveiters=50)
    plt.plot(np.arange(len(train_error)), train_error)
    plt.yscale('log')
    plt.show()

def testRNN():
    x = T.matrix('input')
    y = T.matrix('output')
    rnn = RecurrentLayer(1000, 1000, init_size=0.1, in_var=x, 
            nonlinearity=lambda x: x)

    data = convertMusicFile(0, inputsize=1000)
    err = T.mean(T.sqr(rnn.out - y))

def testAutoEncoder():
    theano.config.floatX ='float32'
    data = convertMusicFile(0)
    ae = AutoEncoder(1000, 800, in_type='linear', init_size=0.1)

    scaleFactor, data = normalize(data)
    y = T.matrix()
    mse = T.mean(T.sqr((ae.reconstructed- y)))

    (rprop, rupdates) = generateRpropUpdates(ae.params, mse, init_size=0.1,
            verbose=False)

    (stor, adam) = generateAdam(ae.params, mse, alpha=0.001)
    (stora, rms) = generateRmsProp(ae.params, mse, alpha=0.0001)
    (storag, momentum) = generateMomentumUpdates(ae.params, mse, alpha=0.001,
            momentum=0.9)

    sgd = generateVanillaUpdates(ae.params, mse, alpha=0.001)

    learn = theano.function([ae.x, y], mse, updates=rupdates,
            allow_input_downcast=True)

    train_error = miniBatchLearning(data[:1000], data[:1000], -1, learn, verbose=True,
            epochs=1000)

    plt.plot(np.arange(len(train_error)), train_error)
    plt.yscale('log')
    plt.show()
    predict = theano.function([ae.x], ae.reconstructed,
            allow_input_downcast=True)
    generateMusicFile(scaleBack(predict(data[:1000]), scaleFactor),
        open("test.wav", "wb"))

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

def testConvNet():
    data = convertMusicFile(0)
    x = data[:1000].reshape(1000, 1, 1, 1000)

    scaleFactor, x = normalize(x)
    print(x.shape, data[:1000].shape)

    conv1 = ConvolutionLayer((1, 1, 1, 10), init_size=0.1,
            nonlinearity = lambda x: x)
    conv2 = ConvolutionLayer((1, 1, 1, 10), in_var = conv1.out,
                nonlinearity = lambda x: x, init_size=0.1, deconv=True)

    out = conv2.out
    y = T.tensor4('target')
    params = conv1.params + conv2.params
    error = T.mean(T.sqr(y - out))
    predict = theano.function([conv1.x], conv2.out)

    (storage, rprop) = generateRpropUpdates(params, error, init_size=0.01)

    learn = theano.function([conv1.x, y], error, updates=rprop)

    train_error = miniBatchLearning(x, x, -1, learn, verbose=True, epochs=10)

    plt.plot(np.arange(len(train_error)), train_error)
    plt.yscale('log')
    plt.show()

    finalP = predict(x)
    print(finalP.min(), finalP.max())
    print(x.min(), x.max())



if __name__ == '__main__':
    testLSTM()
