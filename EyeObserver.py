from DeepLearning import *
import random
import cv2
import tensorflow as tf

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

if __name__=='__main__':
    eyeObserver()
