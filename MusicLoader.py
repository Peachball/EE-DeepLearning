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
import random

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

    output = np.zeros((data.size * 2 // chunk_size, chunk_size))

    row_counter = 0
    for i in range(0, data.shape[0], chunk_size // (2 * channels)):
        row = []
        for j in range(channels):
            comp = np.fft.fft(data[i:i + (chunk_size // (2 * channels)), j])
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
    lstm = LSTM(1024, 800, 800, 1024, verbose=True, init_size=0.1, out_type='linear')


    x = data[:-1]
    y = data[1:]
    # y = x

    # (rprop, rupdates) = generateRpropUpdates(lstm.params, lstm.error,
            # init_size=0.1, verbose=True)
    # (adamstorage, adam) = generateAdam(lstm.params, lstm.error, alpha=0.01,
            # verbose=True)
    (storage , rms) = generateRmsProp(lstm.params, lstm.error, alpha=0.01,
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

    def load_newfile(filename):
        samplerate, data = wavUtil.read(filename)
        _, x = normalize(wav_to_FT(data), scaleFactor=scale)

        print(x.shape)
        reconstructed = lstm.predict(x)
        output = FT_to_wav(scaleBack(reconstructed, scale))

        generateMusicFile(output, open(filename + ".gen.wav", "wb"))


    test()
    try:
        load()
        # load_newfile("testsound.wav")
        # print("loaded testsound too")
    except Exception as e:
        print(e)
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
    scale, data = get_data(0)
    rbm = RBMLayer(1024, 900)

    y = T.matrix()

    persistent = theano.shared(np.zeros((100, 1024)).astype(theano.config.floatX))
    adj_cost, grad_updates = rbm.cost_updates(lr=0.001, persistent=persistent, k=1)

    mse = T.mean(T.sum(T.sqr(rbm.mean_vhv(rbm.x) - y), axis=1))

    learn = theano.function([rbm.x, y], mse, updates=grad_updates,
            allow_input_downcast=True)

    reconstruct = theano.function([rbm.x], rbm.mean_vhv(rbm.x),
            allow_input_downcast=True)

    train_error = miniBatchLearning(data, data, 250, learn, verbose=True,
            epochs=10)
    print("generating files")
    generateMusicFile(FT_to_wav(scaleBack(data, scale)),
            'original.wav')
    generateMusicFile(FT_to_wav(scaleBack(reconstruct(data), scale)),
            'test.wav')
    plt.plot(np.arange(len(train_error)), train_error)
    plt.show()

def testCWRNN():
    scale, data = get_data(0)
    data.astype('float32')
    cw = CWLayer(1024, 1024, 1024, 12, nonlinearity=lambda x:x)

    predict = theano.function([cw.x], cw.out, updates=cw.updates,
            allow_input_downcast=True)

    y = T.matrix()
    error = T.mean(T.sum(T.sqr(cw.out - y), axis=1))

    (storage, grad_updates) = generateAdagrad(cw.params, error, alpha=0.01)

    learn = theano.function([cw.x, y], error, updates=grad_updates,
            allow_input_downcast=True)
    reset = cw.reset

    train_error = miniRecurrentLearning(data, data, 100, learn, predict, reset,
            verbose=True, strides=10)

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

def create_examples(x, length=128, examples=None, uniform=False):

    if examples is None:
        examples = x.shape[0] // length
    if examples > x.shape[0]:
        examples = x.shape[0]

    x_out = np.zeros((examples, length, x.shape[1]))
    y_out = np.zeros((examples, x.shape[1]))

    count = 0
    if not uniform:
        indexes = random.sample(range(x.shape[0] - length, examples))
    if uniform:
        indexes = range(0, x.shape[0] - length, length)[:examples]

    for i in indexes:
        x_out[count] = x[i:(i+length)]
        y_out[count] = x[i+length+1]
        count += 1
    return x_out, y_out

def testKerasLSTM():
    #Load data
    orgdata = convertMusicFile(0)

    orgdata = orgdata.astype('float32')
    scale, data = normalize(wav_to_FT(orgdata))

    #Build lstm model
    from keras.models import Sequential
    from keras.layers import LSTM, Dense, TimeDistributed
    from keras.optimizers import RMSprop

    model = Sequential()
    model.add(TimeDistributed(Dense(1024), input_shape=(128, 1024)))
    model.add(LSTM(1024, return_sequences=True))
    model.add(LSTM(1024, return_sequences=False))

    model.compile(RMSprop(lr=0.001), 'mse')


    def generateSample(length=150, seed=None):
        if seed is None:
            seed = data[:128]
        song = seed

        while song.shape[0] < length:
            y = model.predict(seed[None,-128:])
            song = np.concatenate((song, y), axis=0)
        song = np.squeeze(song)

        generateMusicFile(FT_to_wav(scaleBack(song, scale)),
                open('test' + str(generateSample.count) + '.wav', 'wb'))
        generateSample.count += 1

    generateSample.count = 0

    try:
        model.load_weights("keras_musicgen.h5")
    except:
        print("Unable to load weights")

    while True:
        X_ex, Y_ex = create_examples(data, uniform=True)
        generateSample(length=1024)
        model.fit(X_ex, Y_ex)
        model.save_weights("keras_musicgen.h5", overwrite=True)

def get_data(index):
    od = convertMusicFile(index)
    od = od.astype('float32')

    data = wav_to_FT(od)
    scale, data = normalize(data, low=0, high=1)
    return scale, data

def EEDataGenerator():
    #Build models
    import time
    timefile = open('times.txt', 'a')
    MODE = 'FAST_RUN'

    def construct_model(layers, m_type='lstm'):
        x = T.matrix()
        y = T.matrix()
        updates = None

        if m_type == 'lstm':
            model = LSTM(*((1024,) * (layers + 1)), in_var=x, out_var=y,
                    out_type='linear', init_size=-1)
            params = model.params
            updates = model.updates
            out = model.out
            reset = model.reset

        if m_type == 'rnn':
            model = RNN(*((1024,) * (layers + 1)), in_var=x, out_var=y,
                    out_type='linear', init_size=6)
            out = model.out
            params = model.params
            reset = model.reset

        if m_type == 'cwrnn':
            nonlinearity = T.tanh
            if layers == 1:
                nonlinearity = lambda x:x
            resets = []
            m = CWLayer(1024, 1024, 1024, 16, in_var=x,
                    nonlinearity=nonlinearity)
            updates = m.updates
            resets.append(m.reset)
            params = m.params
            for i in range(layers-1):
                if layers == layers - 2:
                    nonlinearity = lambda x:x
                m = CWLayer(1024, 1024, 1024, 16, in_var=m.out,
                        nonlinearity=nonlinearity)

                updates += m.updates
                resets.append(m.reset)
                params += m.params
            def r():
                for res in resets:
                    res()
                return
            reset = r
            out = m.out

        if m_type == 'gru':
            nonlinearity = T.tanh
            if layers == 1:
                nonlinearity = lambda x:x
            resets = []
            lays = []
            m = GRULayer(1024, 1024, cell_size=1024, in_var=x,
                    nonlinearity=nonlinearity)
            resets.append(m.reset)
            params = m.params
            updates = m.updates
            for i in range(layers-1):
                if layers == layers - 2:
                    nonlinearity = lambda x:x
                m = GRULayer(1024, 1024, cell_size=1024, in_var=m.out,
                        nonlinearity=nonlinearity)

                updates += m.updates
                resets.append(m.reset)
                params += m.params

            def r():
                for re in resets: re()
                return

            reset = r
            reset()
            out = m.out

        if m_type == 'overlapping_lstm':
            pass

        return (x, y, params, out, updates, reset)

    def test_model(x, y, o, params, predict, reset, name):
        error = T.mean(T.sum(T.sqr(y - o), axis=1))

        print("Calculating Gradient Updates...")
        (storage, learn_updates) = generateAdagrad(params, error, alpha=0.001,
                verbose=True)
        print("Compiling Learn Function")
        learn = theano.function([x, y], error, updates=learn_updates,
                mode=MODE, allow_input_downcast=True)

        start_time = time.clock()
        train_error = miniRecurrentLearning(X_dat, Y_dat, 100, learn, predict,
                reset, verbose=True, epochs=5, strides=5)

        duration = time.clock() - start_time

        pickle.dump(train_error, open(name + '.data', 'wb'))

        timefile.write(name + ' took ' + str(duration) + '\n')
        timefile.flush()

    scale, X_dat = get_data(0)
    X_dat = X_dat[:1024]
    print(X_dat.min(), X_dat.max())
    Y_dat = X_dat[:1024]

    #Test RBM

    #Test overlapping LSTMs

    #Test overlapping RNNs


    #Test lstms
    for i in range(1,3):
        print("Constructing " + str(i+1) + " layer lstm")
        x, y, params, o, updates, reset = construct_model(i+1, m_type='lstm')

        predict = theano.function([x], o, updates=updates,
                allow_input_downcast=True)
        test_model(x, y, o, params, predict, reset, str(i+1) + 'LayerLSTM')

    #Test RNNs
    for i in range(3):
        print("Constructing " + str(i+1) + " layer rnn")
        x, y, params, o, updates, reset = construct_model(i+1, m_type='rnn')
        predict = theano.function([x], o, updates=updates,
                allow_input_downcast=True)

        test_model(x, y, o, params, predict, reset, str(i+1) + 'LayerRNN')

    #Test GRUS
    for i in range(3):
        print("Constructing " + str(i+1) + " layer GRU")
        x, y, params, o, updates, reset = construct_model(i+1, m_type='gru')
        predict = theano.function([x], updates=updates,
                allow_input_downcast=True)

        test_model(x, y, o, params, predict, reset, str(i+1) + 'LayerGRU')

    #Test CW
    for i in range(3):
        print("Constructing " + str(i+1) + " layer cw-rnn")
        x, y, params, o, updates, reset = construct_model(i+1, m_type='cwrnn')
        predict = theano.function([x], updates=updates,
                allow_input_downcast=True)

        test_model(x, y, o, params, predict, reset, str(i+1) + 'LayerCWRNN')

if __name__ == '__main__':
    EEDataGenerator()
