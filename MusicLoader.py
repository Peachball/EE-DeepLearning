from __future__ import print_function
import pickle
import theano
from models.DeepLearning import *
from models.RecurrentNetworks import LSTM, GRU
from models.RecurrentNetworks import miniRecurrentLearning
import numpy as np
import scipy.io.wavfile as wavUtil
import matplotlib.pyplot as plt
import scipy.fftpack
import warnings
import random
from os.path import join

SAMPLE_RATE = 44100

def downloadMusicPlaylist(outdir, playlist_url, num=50):
    import youtube_dl
    # for i in range(num):
    ydl_opts = {'format': 'bestaudio/best',
                'postprocessors': [{
                        'key': 'FFmpegExtractAudio',
                        'preferredcodec': 'wav',
                    }],
                'outtmpl' : outdir + '%(playlist_index)s.$(ext)s',
                'playlist_items': num
                }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([playlist_url])

def convertMusicFile(index, inputsize=1000):
    INDEX_NUM = 268
    index = index % INDEX_NUM
    filename = join('datasets','ncs', "{0:0=3d}".format(index) + '.wav')
    samplerate, data = wavUtil.read(filename)

    #Get rid of quiet parts in the beginning and end
    index1 = 0
    index2 = -1
    while data[index1][0] == 0 and data[index1][1] == 0:
        index1 += 1
    while data[index2][0] == 0 and data[index2][1] == 0:
        index2 -= 1
    data = data[index1:(index2)]

    return data

def generateMusicFile(arr, f, channels=2):
    data = arr.reshape(arr.size//channels, channels).astype('int16')
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
    output = np.zeros((data.size // (2 * channels), channels))

    time = 0

    chunk_time_size = data.shape[1] / (2 * channels)
    for row in data:
        time_length = data.shape[1] // channels

        curchannel = 0
        for c in range(0, data.shape[1], time_length):
            ft = row[c:(c+time_length//2)] + 1j * \
                row[(c+time_length//2):(c+time_length)]
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
    scale, data = normalize(wav_to_FT(orgdata), type='gauss')

    #Build lstm model
    from keras.models import Sequential
    from keras.layers import LSTM, Dense, TimeDistributed, GRU
    from keras.optimizers import RMSprop

    model = Sequential()
    model.add(GRU(1024, return_sequences=True, batch_input_shape=(1, 1, 1024),
        stateful=True))
    model.add(GRU(1024, return_sequences=True, stateful=True))

    model.compile(RMSprop(lr=0.001), 'mse')


    def generateSample(length=150, seed=None):
        if seed is None:
            seed = data[:1]
        song = seed

        print("Generating song...")
        while song.shape[0] < length:
            y = model.predict(song[None,-1:])
            song = np.concatenate((song, y[:,0,:]), axis=0)
        song = np.squeeze(song)

        generateMusicFile(FT_to_wav(scaleBack(song, scale, type='gauss')),
                open('test' + str(generateSample.count) + '.wav', 'wb'))
        generateSample.count += 1

        print("Done with song")

    generateSample.count = 0

    try:
        model.load_weights("keras_musicgen.h5")
    except:
        print("Unable to load weights")

    while True:
        X_ex, Y_ex = data[:-1], data[1:]
        model.reset_states()
        generateSample(length=150)
        model.reset_states()
        for i in range(X_ex.shape[0]):
            e = model.train_on_batch(X_ex[None,i:i+1,:], Y_ex[None,i:i+1,:])
            print("Iteration: {0:.4f} Error: {1}".format(i+1, e))
        model.save_weights("keras_musicgen.h5", overwrite=True)

def testtfLSTM():
    import tensorflow as tf
    X = tf.placeholder(tf.float32, [1, 2, 2048], name='input')
    Y = tf.placeholder(tf.float32, [1, 2, 2048], name='label')

    with tf.variable_scope("LSTM") as scope:
        lstm = tf.nn.rnn_cell.LSTMCell(2048, use_peepholes=True)
        stacked = tf.nn.rnn_cell.MultiRNNCell([lstm] * 4)
        init_state = stacked.zero_state(1, tf.float32)

    def add_transformation(inp, out_size, name):
        shape = [int(s) for s in inp.get_shape()]
        with tf.variable_scope(name) as scope:
            w = tf.get_variable('w', shape=[inp.get_shape()[-1], out_size],
                    initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable('b', shape=[out_size])
            scope.reuse_variables()
        if len(inp.get_shape()) == 2:
            return tf.matmul(inp, w) + b
        else:
            downcast = tf.reshape(inp, (-1, int(inp.get_shape()[2])))
            mult = tf.matmul(downcast, w) + b

            return tf.reshape(tf.matmul(downcast, w) + b, (inp.get_shape()))

    num_steps = 2
    outputs, final_state = tf.nn.dynamic_rnn(stacked, inputs=X,
            initial_state=init_state)
    tf.histogram_summary('state', final_state)
    tf.histogram_summary('outputs', outputs)
    gen = add_transformation(outputs, 2048, 'out_transform')

    loss = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(Y, gen), 2))
    tf.scalar_summary('loss', loss)

    train_step = tf.train.RMSPropOptimizer(0.01).minimize(loss)
    merged = tf.merge_all_summaries()
    sw = tf.train.SummaryWriter('tflogs/musicgen')

    init = tf.initialize_all_variables()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        for d in data_generator():
            s = sess.run(init_state)
            for i in range(num_steps, d.shape[0], num_steps):
                x = d[None,i-num_steps:i,:]
                y = d[None,i:i+num_steps,:]
                _, s, e, summaries = sess.run(
                        [train_step, final_state, loss, merged],
                        feed_dict={X: x, Y: y, init_state: s})
                saver.save(sess, 'tflogs/musicgen/model.cpkt')
                sw.add_summary(summaries)
                print(e)


def get_data(index, channels='reg', chunk_size=1024, scale=None):
    od = convertMusicFile(index)
    od = od.astype('float32')

    if channels == 'mono':
        od = od[:,:-1]
    data = wav_to_FT(od, chunk_size=chunk_size)
    if scale is None:
        scale, data = normalize(data, type='gauss')
    else:
        _, data = normalize(data, type='gauss', scaleFactor=scale)
    return scale, data

def trainGRU():
    CHANNELS = 1
    INPUT_SIZE = 2048
    lstm = GRU(*(4 * (INPUT_SIZE,)), nonlinearity=lambda x:x,
            init_size=0.001, verbose=True)
    y_ = T.matrix()
    error = T.mean(T.sum(T.sqr(lstm.out - y_), axis=1))
    (storage, upd) = generateRmsProp(lstm.params, error, alpha=1e-4, verbose=True)

    predict = lstm.predict
    reset = lstm.reset

    print("Loading data")
    scale, X_dat = get_data(1, 'mono', chunk_size=INPUT_SIZE)

    error_file = open('rundata/gausslstmerror.txt', 'a')

    model_name = 'gaussbiglstm'
    def save():
        saveParams(lstm.params, 'rundata/' + model_name)
        generate(name='rundata/' + str(save.filename)+'test_boring.wav')
        generate(
                name='rundata/' + str(save.filename)+'test_interesting.wav',
                lame=False)
        save.filename += 1
        return

    save.filename = 0

    def load():
        loadParams(lstm.params, 'rundata/' + model_name + '.npz')
        return

    def generate(lame=True, name='test.wav'):
        if lame:
            song = predict(X_dat[:1000])
            generateMusicFile(FT_to_wav(scaleBack(song, scale, type='gauss'),
                channels=CHANNELS) , name, channels=1)
        else:
            length = 1000
            song = np.zeros((length, INPUT_SIZE)).astype('float32')
            song[:1] = X_dat[:1]
            for i in range(1, length):
                song[i:i+1] = predict(song[i-1:i])
                print("\r{} of the way there".format(i/length), end="")
            generateMusicFile(FT_to_wav(scaleBack(song, scale, type='gauss'),
                channels=CHANNELS), name, channels=CHANNELS)
            print("")
        return

    try:
        load()
    except:
        print("Failed to load previous model")

    print("Compiling learn function")
    learn = theano.function([lstm.x, y_], error, updates=upd,
            allow_input_downcast=True)
    print("Commencing learning")
    train_error = []
    for d in data_generator(scale=scale):
        train_error += miniRecurrentLearning(d[:-1], d[1:], 50, learn, predict,
                reset,
                verbose=True, epochs=1, save=save, saveiters=300, strides=50,
                f=error_file)
        save()

    pickle.dump(train_error, open('lstm_train.data', 'wb'))

def data_generator(chunk_size=2048, scale=None):
    i = 0
    try:
        with open('rundata/train.meta', 'r') as f:
            i = int(f.read())
    except:
        print("Failed to load previous file location")

    s, d = get_data(i, channels='mono', scale=scale, chunk_size=chunk_size)
    yield d
    i = 1
    while True:
        _, d = get_data(i, channels='mono', scale=s, chunk_size=chunk_size)
        yield d
        i += 1

        with open('rundata/train.meta', 'w') as f:
            f.write(i)

def EEDataGenerator():
    #Build models
    import time
    from os.path import join
    DATA_DIR = join('rundata', 'times.txt')
    timefile = open(DATA_DIR, 'a')
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

        pickle.dump(train_error, open(join('data',name + '.data'), 'wb'))

        timefile.write(name + ' took ' + str(duration) + '\n')
        timefile.flush()

    scale, X_dat = get_data(0)
    X_dat = X_dat[:1024]
    print(X_dat.min(), X_dat.max())
    Y_dat = X_dat[:1024]

    #Test lstms
    for i in range(3):
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
    trainGRU()
