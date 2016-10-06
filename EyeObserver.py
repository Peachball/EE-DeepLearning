from __future__ import print_function
import argparse
import sys
from DeepLearning import *
import random
import cv2
import tensorflow as tf
import os
import random
import matplotlib.pyplot as plt
from keras.layers import Convolution2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.models import Model, Sequential, model_from_yaml
from keras.layers import BatchNormalization
from keras.optimizers import SGD, RMSprop, Adam
import h5py

config = {
        "image_path" : "eyesDataSet",
        "image_dimension" : (304, 171)
        }

def readFile(f, see=True, start=0, limit=1e10, background=False):
    if see:
        org = './eyesDataSet/see/'
    else:
        org = './eyesDataSet/nosee/'
    if background:
        org = './eyesDataSet/noeye/'

    vidcap = cv2.VideoCapture(org + f)
    success, image = vidcap.read()
    frame = start
    images = []
    while success:
        images.append(image)
        frame += 1
        if frame + start > limit:
            return images
        success, image = vidcap.read()
    return images

def getData(num):
    totalimages  = 1
    images = []
    labels = []

    for i in range(num+1):
        #Read looking at screen videos
        new_images = readFile(str(i), see=True)
        images = images + new_images
        labels = labels + (len(new_images) * [1])

        #Read looking away videos
        new_images = readFile(str(i), see=False)
        images = images + new_images
        labels = labels + (len(new_images) * [0])
    data = zip(images, labels)
    random.shuffle(data)
    return data

def eyeObserver():
    from theano.tensor.signal import pool
    im_width = 1080
    im_height = 1920
    x = T.tensor4("Input")
    y = T.matrix("Labels")

    data = getData(0)
    data = data[:10]
    [images, labels] = zip(*data)
    images = np.array(images)
    labels = np.array(labels)

    images = images.transpose(0, 3, 2, 1)
    labels = labels.reshape(10, 1)

    print("Building Convolution Layers")
    conv1 = ConvolutionLayer((8, 3, 20, 20), in_var=x, init_size=0.1,
            subsample=(4,4))
    conv2 = ConvolutionLayer((6, 8, 15, 15), in_var=conv1.out, init_size=0.1,
            subsample=(4,4))
    conv3 = ConvolutionLayer((4, 6, 10, 10), in_var=conv2.out, init_size=0.1,
            subsample=(4,4))
    conv4 = ConvolutionLayer((2, 4, 5, 5), in_var=conv3.out, init_size=0.1,
            subsample=(1,1))

    print("Finished Building Convoution Layers")

    intermediate = T.flatten(conv4.out, outdim=2)

    test = theano.function([x], outputs=[intermediate])

    class1 = Layer(460, 200, in_var=intermediate, layer_type='rlu',
            init_size=0.1)
    class2 = Layer(200, 1, in_var=class1.out, layer_type='sigmoid',
            init_size=0.1)

    output = class2.out
    prediction = theano.function([x], output)

    error = -T.mean(y * T.log(output) + (1-y) * T.log(1 - output))

    params = conv1.params + conv2.params + conv3.params + conv4.params + \
        class1.params + class2.params

    (storage, rprop) = generateRpropUpdates(params, error, init_size=0.01,
            verbose=True)
    sgd = generateVanillaUpdates(params, error, alpha=0.01)
    (rmsStorage, rms) = generateRmsProp(params, error, alpha=0.003)

    filestorage = params + [i for l in rmsStorage for i in l]
    print("Compiling learn function")
    learn = theano.function([x, y], error, updates=rprop)

    try:
        print("Loading previous paramaters")
        loadParams(filestorage, "eyeTrack.conv.npz")
    except Exception as e:
        print("Unable to load previous paramaters")

    miniBatchLearning(images, labels, -1, learn, verbose=True, epochs=10)

    saveParams(filestorage, "eyeTrack.conv")

def setup_data(directory=config["image_path"]):
    """
        Convert all pictures to videos in the dir, and put them in a new folder
    """
    from os.path import join
    subdirs = ['noeye', 'nosee', 'see']

    for sd in subdirs:
        subdir_name = os.path.join(directory, sd)

        #Create dir for images
        image_file = os.path.join(subdir_name, 'images')
        if not os.path.exists(image_file):
            os.makedirs(image_file)
        count = len(os.listdir(image_file))

        for f in os.listdir(subdir_name):
            if os.path.isdir(join(subdir_name, f)):
                continue
            video_capture = cv2.VideoCapture(os.path.join(subdir_name, f))
            while True:
                ret, frame = video_capture.read()
                if frame is None: break

                image_name = os.path.join(image_file, str(count) + '.png')
                cv2.imwrite(image_name, frame)
                count += 1
            os.rename(join(subdir_name, f), join(subdir_name, 'done', f))

def get_data(noeye=0, nosee=0, see=0, verbose=False):
    X_dat = []
    label = []

    if verbose:
        total_files = len(os.listdir('eyesDataSet/noeye/images'))
        total_files += len(os.listdir('eyesDataSet/nosee/images'))
        total_files += len(os.listdir('eyesDataSet/see/images'))
        file_count = 0
        v_info = {'file_count': file_count,
                  'total': total_files}
    def get_data_from_dir(amount, directory, size=config["image_dimension"]):
        from PIL import Image
        data = []
        count = 0

        usable_files = range(len(os.listdir(directory)))
        if amount > len(usable_files) or amount < 0:
            amount = len(usable_files)

        usable_files = random.sample(usable_files, amount)

        for n in usable_files:
            filename = os.path.join(directory, str(n) + '.png')
            # data.append(cv2.imread(filename))
            img = Image.open(filename)
            img.thumbnail(size)

            data.append(np.array(img))
            if verbose:
                file_count = v_info['file_count']
                total = v_info['total']
                print("\r{}/{} files loaded".format(file_count, total_files), end="")
                sys.stdout.flush()
                v_info['file_count'] = file_count + 1

        return data

    noeye_dat = get_data_from_dir(noeye,
            os.path.join('eyesDataSet', 'noeye', 'images'))

    nosee_dat = get_data_from_dir(nosee,
            os.path.join('eyesDataSet', 'nosee', 'images'))

    see_dat = get_data_from_dir(see,
            os.path.join('eyesDataSet', 'see', 'images'))


    Y_dat = [[1, 0, 0]] * len(noeye_dat) + [[0, 1, 0]] * len(nosee_dat) +\
            [[0, 0, 1]] * len(see_dat)
    if verbose:
        print("")

    return np.array(noeye_dat + nosee_dat + see_dat), np.array(Y_dat)

def KerasEyeObserver():
    print("Loading data")
    X_dat, Y_dat = get_data(noeye=-1, nosee=-1, see=-1, verbose=True)
    print("Done loading data")
    X_dat = X_dat.astype('float32')
    X = X_dat.transpose(0, 3, 1, 2) / 256

    # while True:
        # import random
        # count = random.randint(0, X.shape[0] - 1)
        # plt.imshow(X.transpose(0, 2, 3, 1)[count] * 256 + 128, interpolation='none')
        # plt.figure()
        # plt.imshow((255 - X_dat[count]), interpolation='none')
        # plt.show()

    act = 'softplus'

    #Generate model (Based off of VGGs ImageNet)
    print("Generating model")
    model = Sequential()

    model.add(Convolution2D(48, 11, 11, activation=act, border_mode='same',
                        input_shape=(3,)+config['image_dimension'][::-1],
                        subsample=(4,4)))
    model.add(MaxPooling2D((2, 2), border_mode='same'))
    model.add(BatchNormalization(axis=1))
    model.add(Dropout(0.5))
    model.add(Convolution2D(128, 5, 5, activation=act,
        border_mode='same'))
    model.add(MaxPooling2D((2, 2), border_mode='same'))


    model.add(BatchNormalization(axis=1))
    model.add(Dropout(0.5))
    model.add(Convolution2D(192, 3, 3, activation=act,
        border_mode='same'))
    model.add(BatchNormalization(axis=1))
    model.add(Dropout(0.5))
    model.add(Convolution2D(192, 3, 3, activation=act,
        border_mode='same'))
    model.add(BatchNormalization(axis=1))
    model.add(Dropout(0.5))
    model.add(Convolution2D(128, 3, 3, activation=act,
        border_mode='same'))
    model.add(MaxPooling2D((2, 2), border_mode='same'))

    model.add(Flatten())

    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(2048, activation='tanh'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(2048, activation='tanh'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))

    print("Compiling model")
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    rms = RMSprop(lr=0.0001)
    model.compile(optimizer=sgd, loss='categorical_crossentropy',
                    metrics=['accuracy'])

    # try:
        # model.load_weights("eyeobserver.h5")
    # except Exception as e:
        # print(e)

    model.save("eyeobserver.h5", True)

    for i in range(10):
        model.fit(X, Y_dat, batch_size=32, validation_split=0.2,
                nb_epoch=2)

        model.save("eyeobserver.h5", True)

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.legend(['train', 'test'])
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['train', 'test'])
    plt.show()
    print(history.history.keys())

def tfEyeObserver():
    X_dat, Y_dat = get_data(noeye=-1, nosee=-1, see=-1)
    X_dat = X_dat.astype('float32') / 128 - 1

    X_dat = X_dat.transpose(0, 2, 1, 3)

    x = tf.placeholder(tf.float32,
            shape=[None]+list(config['image_dimension']) + [3])
    y_ = tf.placeholder(tf.float32, shape=[None, 3])
    tf.image_summary("Input images", x)

    xavier = tf.contrib.layers.xavier_initializer()
    xavier2d = tf.contrib.layers.xavier_initializer_conv2d()
    uniform_init = tf.random_uniform_initializer(minval=-0.05, maxval=0.05)

    def add_conv_layer(inp, scope, shape):
        with tf.variable_scope(scope):
            filt = tf.get_variable("filter", shape=shape,
                    initializer=xavier2d)
            bias = tf.get_variable("bias", shape=[shape[3]],
                    initializer=uniform_init)

            layer = tf.nn.relu(tf.nn.conv2d(inp, filt, [1, 1, 1, 1], "SAME") + bias)
        return layer

    def add_dense_layer(inp, scope, shape, activation=tf.nn.relu):
        with tf.variable_scope(scope):
            weight = tf.get_variable("weight", shape=shape, initializer=xavier)
            bias = tf.get_variable('bias', shape=[shape[1]],
                    initializer=uniform_init)
            layer = activation(tf.matmul(inp, weight) + bias)
        return layer

    layer1 = add_conv_layer(x, "conv1", [11, 11, 3, 96])

    layer2_1 = tf.nn.max_pool(layer1, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")

    layer2 = add_conv_layer(layer2_1, "conv2", [5, 5, 96, 256])

    layer3 = add_conv_layer(layer2, "conv3", [3, 3, 256, 192])

    layer4_1 = tf.nn.max_pool(layer3, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")

    layer4_2 = tf.reshape(layer4_1, [-1, 176640], name='layer4')

    layer5 = add_dense_layer(layer4_2, "dense5", [176640, 4096]
            , activation=tf.tanh)
    layer6 = add_dense_layer(layer5, "dense6", [4096, 4096],
            activation=tf.tanh)

    layer7 = add_dense_layer(layer5, "dense7", [4096, 3],
            activation=tf.nn.softmax)

    out = layer7

    cross_entropy = tf.reduce_mean(tf.reduce_sum(-y_ * tf.log(out), 1))

    tf.scalar_summary("Loss over time", cross_entropy)

    num_correct = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(y_, 1),
        tf.argmax(out, 1)), tf.float32))

    train_op = tf.train.AdamOptimizer().minimize(cross_entropy)

    init_op = tf.initialize_all_variables()
    summary = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter('./tfeyetrainer/debug')
    saver = tf.train.Saver()

    sv = tf.train.Supervisor(init_op=init_op,
            logdir='./tfeyetrainer/models',
            saver=saver,
            summary_op=summary)

    with sv.managed_session() as sess:
        while True:
            for i in range(0, X_dat.shape[0], 3):
                _, loss, cor, summary = sess.run([out, cross_entropy,
                                                  num_correct, summary_op],
                                    feed_dict={x: X_dat[:5], y_:Y_dat[:5]})

            writer.add_summary(summary)
            print(loss, cor)

def real_time_prediction():
    import Tkinter as tk
    from Tkinter import StringVar
    from keras.models import load_model, model_from_yaml
    from PIL import ImageTk, Image

    model = load_model('eyeobserver.h5')
    data = {}
    # model = model_from_yaml('eyeobserver.model')
    # model.load_weights('eyeobserver.h5')
    size = config['image_dimension']
    cap = cv2.VideoCapture(0)
    root = tk.Tk()
    root.bind('<Escape>', lambda e: root.quit())
    ptext = StringVar()
    ftext = StringVar()
    ptext.set("hi")

    lmain = tk.Label(root)
    p = tk.Label(root, text=ptext)
    finalprediction = tk.Label(root, text=ftext)


    p.grid(row=0, column=0)
    finalprediction.grid(row=1, column=0)
    lmain.grid(row=3, column=0)

    def showImage():
        if not showImage.freeze:
            _, img = cap.read()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
            img = img[:,:,:3]
            img = cv2.resize(img, config['image_dimension'])
            showImage.image = img
            imgtk = ImageTk.PhotoImage(
                    image=Image.fromarray(img).resize((1280, 720), Image.BICUBIC))
            lmain.imgtk = imgtk
            lmain.configure(image=imgtk)

        prediction = model.predict(
                showImage.image[None,:,:,:].transpose(0, 3, 1, 2) / 255)
        pstr = ("No eye likelihood: {0}\n"
                "Not looking likelihood: {1}\n"
                "Looking likelihood: {2}").format(prediction[0,0],
                                                  prediction[0,1],
                                                  prediction[0,2])
        answer = ""
        a = np.argmax(prediction[0])
        if a == 0:
            answer = "There isn't an eye here"
        if a == 1:
            answer = "The eye is not looking"
        if a == 2:
            answer = "The eye is looking"

        p.configure(text=pstr)
        finalprediction.configure(text=answer)

        lmain.after(10, showImage)
    showImage.freeze = False
    showImage.image = None
    showImage.counter = 0

    tk.Button(root, text='Save noeye', command=
            lambda: saveImage(Image.fromarray(showImage.image), 0)).grid(
                    row=0, column=1)
    tk.Button(root, text='Save nosee', command=
            lambda: saveImage(Image.fromarray(showImage.image), 1)).grid(
                    row=1, column=1)
    tk.Button(root, text='Save see', command=
            lambda: saveImage(Image.fromarray(showImage.image), 2)).grid(
                    row=2, column=1)

    def freeze(e):
        showImage.freeze = not showImage.freeze

    def saveImage(img, category):
        dirname = 'eyesDataSet/'
        if category == 0:
            dirname += 'noeye/'
        if category == 1:
            dirname += 'nosee/'
        if category == 2:
            dirname += 'see/'
        dirname += 'images/'
        from os import listdir
        from os.path import isdir, join

        files = [f for f in listdir(dirname) if not isdir(join(dirname, f))]
        name = str(len(files)) + '.png'
        # img.save(join(dirname, name))

        if not 'x' in data:
            img = np.array(img)
            images = img.transpose(2, 0, 1).astype('float32') / 255
            data['x'] = images[None, :, :, :]
            data['y'] = np.zeros((1, 3))
            data['y'][0,category] = 1
        else:
            img = np.array(img)
            fixed_image = img.transpose(2, 0, 1).astype('float32') / 255
            fixed_image = fixed_image[None, :, :, :]
            l = np.zeros((1, 3))
            l[0,category] = 1
            data['y'] = np.concatenate((data['y'], l), axis=0)
            data['x'] = np.concatenate((data['x'], fixed_image), axis=0)

    data['training'] = False
    def nonstopTrain():
        data['training'] = True
        while data['training']:
            train(None)

    def startTrain(e):
        import threading
        t = threading.Thread(target=nonstopTrain)
        t.daemon = True
        t.start()

    def stopTrain(e):
        data['training'] = False

    def train(e):
        if not 'x' in data:
            return
        model.fit(data['x'], data['y'], nb_epoch=1)

    root.bind('p', freeze)
    root.bind('t', train)
    root.bind('r', startTrain)
    root.bind('s', stopTrain)

    showImage()
    root.mainloop()

if __name__=='__main__':
    DESCRIPTION = "Toolkit for training + testing an eye tracker"
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('mode')

    args = parser.parse_args()
    if args.mode == "train":
        KerasEyeObserver()
    if args.mode == "predict":
        real_time_prediction()
    if args.mode == "hybrid":
        pass
