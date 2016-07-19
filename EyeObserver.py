from __future__ import print_function
from DeepLearning import *
import random
import cv2
import tensorflow as tf
import os
import random
import matplotlib.pyplot as plt
from keras.layers import Convolution2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.models import Model, Sequential, model_from_yaml
from keras.optimizers import SGD, RMSprop
import h5py

config = {
        "image_path" : "eyesDataSet",
        "image_dimension" : (90, 160)
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
    subdirs = ['noeye', 'nosee', 'see']

    for sd in subdirs:
        count = 0
        subdir_name = os.path.join(directory, sd)

        #Create dir for images
        image_file = os.path.join(subdir_name, 'images')
        if os.path.exists(image_file):
            import shutil
            shutil.rmtree(image_file)
        os.makedirs(image_file)

        for f in os.listdir(subdir_name):
            video_capture = cv2.VideoCapture(os.path.join(subdir_name, f))
            while True:
                ret, frame = video_capture.read()
                if frame is None: break

                image_name = os.path.join(image_file, str(count) + '.png')
                cv2.imwrite(image_name, frame)
                count += 1

def get_data(noeye=0, nosee=0, see=0):
    X_dat = []
    label = []

    def get_data_from_dir(amount, directory, size=config["image_dimension"]):
        from PIL import Image
        data = []
        count = 0

        usable_files = range(len(os.listdir(directory)))
        if amount > len(usable_files):
            amount = len(usable_files)

        usable_files = random.sample(usable_files, amount)

        for n in usable_files:
            filename = os.path.join(directory, str(n) + '.png')
            # data.append(cv2.imread(filename))
            img = Image.open(filename)
            img.thumbnail(size)

            data.append(np.array(img))

        return data

    noeye_dat = get_data_from_dir(noeye,
            os.path.join('eyesDataSet', 'noeye', 'images'))

    nosee_dat = get_data_from_dir(nosee,
            os.path.join('eyesDataSet', 'nosee', 'images'))

    see_dat = get_data_from_dir(see,
            os.path.join('eyesDataSet', 'see', 'images'))


    Y_dat = [[1, 0, 0]] * len(noeye_dat) + [[0, 1, 0]] * len(nosee_dat) +\
            [[0, 0, 1]] * len(see_dat)

    return np.array(noeye_dat + nosee_dat + see_dat), np.array(Y_dat)


def KerasEyeObserver():
    X_dat, Y_dat = get_data(noeye=200, nosee=0, see=200)
    scale = (256, 128)
    _, X = normalize(X_dat.transpose(0, 3, 1, 2), scaleFactor = scale)

    # count = 0
    # while True:
        # plt.imshow(X_dat[count], interpolation='none')
        # plt.show()
        # count+=1

    act = 'relu'

    #Generate model (Based off of VGGs ImageNet)
    print("Generating model")
    model = Sequential()

    model.add(Convolution2D(96, 11, 11, activation=act, border_mode='same',
                        input_shape=(3,)+config['image_dimension'][::-1],
                        subsample=(4, 4)))
    model.add(MaxPooling2D((2, 2), border_mode='same'))
    model.add(Convolution2D(256, 5, 5, activation=act,
        border_mode='same'))


    model.add(Convolution2D(256, 3, 3, activation=act,
        border_mode='same'))
    model.add(Convolution2D(192, 3, 3, activation=act,
        border_mode='same'))
    model.add(MaxPooling2D((2, 2), border_mode='same'))

    # model.add(Convolution2D(384, 3, 3, activation=act,
        # border_mode='same'))
    # model.add(Convolution2D(384, 3, 3, activation=act,
        # border_mode='same'))
    # model.add(Convolution2D(256, 3, 3, activation=act,
        # border_mode='same'))

    model.add(Flatten())

    model.add(Dense(4096, activation='tanh'))
    model.add(Dense(4096, activation='tanh'))
    model.add(Dense(3, activation='softmax'))

    print("Compiling model")
    sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    rms = RMSprop(lr=0.0001)
    model.compile(optimizer=rms, loss='categorical_crossentropy',
                    metrics=['accuracy'])

    try:
        model.load_weights("eyeobserver.h5")
    except Exception as e:
        print(e)

    history = model.fit(X, Y_dat, batch_size=3, validation_split=0.2,
            nb_epoch=20)

    model.save_weights("eyeobserver.h5")

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.legend(['train', 'test'])
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['train', 'test'])
    plt.show()
    print(history.history.keys())


if __name__=='__main__':
    KerasEyeObserver()
