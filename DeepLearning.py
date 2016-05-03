from __future__ import print_function
import sys
import pickle
from collections import OrderedDict
import theano.tensor as T
import theano
from theano.compile.nanguardmode import NanGuardMode
from theano import config
import numpy as np
import struct
import matplotlib.pyplot as plt
import time

sys.setrecursionlimit(10000)

def readMNISTData(length=10000):
	images = open('train-images-idx3-ubyte', 'rb')
	labels = open('train-labels-idx1-ubyte', 'rb')
	images.read(8)
	labels.read(8)
	def readInt(isn=True):
		if isn:
			return struct.unpack('>i', images.read(4))[0]
		else:
			return int.from_bytes(labels.read(4), byteorder='big', signed=True)
	xsize = readInt()
	ysize = readInt()
	def readImage():
		img = []
		for i in range(xsize):
			for j in range(ysize):
				img.append(struct.unpack('>B', images.read(1))[0])
		return img

	def readLabel():
		testLabel = [0]*10
		testLabel[struct.unpack('B', labels.read(1))[0]] = 1
		return testLabel

	imgs = []
	lbls = []
	for i in range(length):
		imgs.append(readImage())
		lbls.append(readLabel())
		print('\rRead {}/{}'.format(i+1, length), end="")
	print('\nDone reading')
	return (np.array(imgs), np.array(lbls))

def readcv(length=10000):
	images = open('t10k-images-idx3-ubyte', 'rb')
	labels = open('t10k-labels-idx1-ubyte', 'rb')
	images.read(8)
	labels.read(8)
	def readInt(isn=True):
		if isn:
			return struct.unpack('>i', images.read(4))[0]
		else:
			return int.from_bytes(labels.read(4), byteorder='big', signed=True)
	xsize = readInt()
	ysize = readInt()
	def readImage():
		img = []
		for i in range(xsize):
			for j in range(ysize):
				img.append(struct.unpack('>B', images.read(1))[0])
		return img

	def readLabel():
		testLabel = [0]*10
		testLabel[struct.unpack('B', labels.read(1))[0]] = 1
		return testLabel

	imgs = []
	lbls = []
	for i in range(length):
		imgs.append(readImage())
		lbls.append(readLabel())
		print('\rRead {}/{}'.format(i+1, length), end="")

	print ('\nDone Reading')
	return (np.array(imgs), np.array(lbls))


class Layer:
	def __init__(self, in_size, out_size, layer_type='sigmoid', in_var=None,
			init_size=0.1, nonlinearity=None):
		self.in_size = in_size
		self.out_size = out_size
		self.nonlinearity = nonlinearity
		if not layer_type in ['sigmoid', 'tanh', 'lstm', 'rnn', 'linear', 'rlu'] and nonlinearity==None:
			raise Exception('Layer type is invalid: ' + str(layer_type))

		if nonlinearity==None:
			if layer_type == 'sigmoid':
				nonlinearity = T.nnet.sigmoid
			if layer_type == 'tanh':
				nonlinearity = T.tanh
			if layer_type == 'linear':
				nonlinearity = lambda x: x
			if layer_type == 'rlu':
				nonlinearity = lambda x: T.log(1 + T.power(np.e, x))

		if in_var==None:
			x = T.matrix('Input')
			print('hello')
		else:
			x = in_var

		self.w = theano.shared((np.random.uniform(low=-init_size, high=init_size, size=(in_size, out_size)).astype(theano.config.floatX) - 0.5) *
				init_size).astype(theano.config.floatX)
		self.b = theano.shared((np.random.rand(out_size).astype(theano.config.floatX) - 0.5) *
				init_size).astype(theano.config.floatX)
		self.out = nonlinearity(T.dot(x, self.w) + self.b )
		self.params = [self.w, self.b]

	def getOutput(self, x,  nonlinearity=T.nnet.sigmoid):
		out = nonlinearity(T.dot(x, self.w) + self.b)
		return out

def reset(params, init_size=0.1):
	for p in params:
		p.set_value(np.random.uniform(low=-init_size, high=init_size, size=p.get_value().shape))

def generateHessianUpdates(params, error, alpha, epsilon=1e-8):
	updates = []

	for p in params:
		shape = p.get_value().shape
		grad = T.grad(error, p)
		flattenedP = T.flatten(p)
		hessian = theano.gradient.hessian(error, flattenedP).diagonal()
		hessian = hessian.reshape(shape)

		updates.append((p, p - grad / (hessian + epsilon) * alpha))

	return updates
def generateAdagrad(params, error, alpha=0.01, epsilon=1e-8):
	updates = []
	history = []

	for p in params:
		shape = p.get_value().shape
		grad = T.grad(error, p)

		totalG = theano.shared(value=np.zeros(shape)).astype(theano.config.floatX)

		new_g = totalG + T.sqr(grad)
		updates.append((totalG, new_g))
		updates.append((p, p - grad / (T.sqrt(new_g) + epsilon) * alpha))

		history.append(totalG)

	return (history, updates)

def generateAdadelta(params, error, decay=0.9, alpha=1, epsilon=1e-8):
	updates = []
	accUpdates = []
	accGrad = []
	for p in params:
		shape = p.get_value().shape
		grad = T.grad(error, p)

		Eg = theano.shared(value=np.zeros(shape)).astype(theano.config.floatX)
		Ex = theano.shared(value=np.zeros(shape)).astype(theano.config.floatX)

		new_g = decay * Eg + (1 - decay) * T.sqr(grad)

		d_x = T.sqrt((Ex + epsilon) / (new_g + epsilon)) * grad * alpha
		new_x = decay * Ex + (1 - decay) * T.sqr(d_x)

		updates.append((p, p - d_x))
		updates.append((Ex, new_x))
		updates.append((Eg, new_g))

		accUpdates.append(Ex)
		accGrad.append(Eg)

	return ([accUpdates, accGrad], updates)

def generateAdam(params, error, alpha=0.001, decay1=0.9, decay2=0.999, epsilon=1e-8, verbose=False):
	updates = []
	moment = []
	vector = []
	alpha = theano.shared(alpha)
	time = theano.shared(value=1.0).astype(theano.config.floatX)
	updates.append((time, time+1))
	i = 0
	for p in params:
		shape = p.get_value().shape
		grad = T.grad(error, p)

		m = theano.shared(value=np.zeros(shape), name="moment").astype(theano.config.floatX)
		v = theano.shared(value=np.zeros(shape), name="moment 0").astype(theano.config.floatX)

		m_t = decay1 * m + (1 - decay1) * grad
		v_t = decay2 * v + (1 - decay2) * T.sqr(grad)
		m_adj = m_t / (1.0 - T.pow(decay1, time))
		v_adj = v_t / (1.0 - T.pow(decay2, time))
		updates.append((m, m_t))
		updates.append((v, v_t))
		updates.append((p, p - alpha * m_adj / (T.sqrt(v_adj) + epsilon)))

		moment.append(m)
		vector.append(v)
		if verbose: print("\rDone with {}/{}".format(i+1, len(params)), end="")
		i += 1

	return ([moment, vector, time, alpha], updates)

def generateRmsProp(params, error, alpha=0.01, decay=0.9, fudge=1e-3):
	r = []
	v = []
	pr = []
	updates = []
	alpha = theano.shared(alpha)
	for p in params:
		grad = T.grad(error, p)

		shape = p.get_value().shape
		r_t = theano.shared(np.zeros(shape)).astype(theano.config.floatX)
		v_t = theano.shared(np.zeros(shape)).astype(theano.config.floatX)

		new_r = (1 - decay) * T.sqr(grad) + decay * r_t
		new_v = alpha / (T.sqrt(new_r) + fudge) * grad
		updates.append((r_t, new_r))
		updates.append((v_t, new_v))
		updates.append((p, p - new_v))
		r.append(r_t)
		v.append(v_t)

	return [[r, v, alpha], updates]
	#lmao not done here

def generateVanillaUpdates(params, error, alpha=0.01):
	grad = []
	for p in params:
		grad.append(T.grad(error, p))
	updates = [(p, p - g * alpha) for p, g in zip(params, grad)]
	return updates

def generateMomentumUpdates(params, error, alpha=0.01, momentum=0.5):
	grad = []
	for p in params:
		grad.append(T.grad(error, p))
	mparams = [theano.shared(np.zeros(p.get_value().shape)).astype(theano.config.floatX) for p in params]
	gradUpdates = OrderedDict((p, p - g) for p, g in zip(params,mparams))

	gradUpdates.update(OrderedDict((m, momentum * m + alpha * g) for m, g in
		zip(mparams, grad)))
	return ([gradUpdates, mparams], gradUpdates)

def generateRpropUpdates(params, error, init_size=1, verbose=False):
	prevw = []
	deltaw = []
	updates = []
	gradients = []
	#initalize stuff
	for p in params:
		prevw.append(theano.shared(np.zeros(p.get_value().shape)).astype(config.floatX))
		deltaw.append(theano.shared(init_size *  np.ones(p.get_value().shape)).astype(config.floatX))

	iterations = 0
	for p, dw, pw in zip(params, deltaw, prevw):
		try:
			if verbose: print("\rGradient {} out of {}".format(iterations + 1, len(params)), end="")
			gradients.append(T.grad(error, p))
			iterations += 1
		except Exception:
			print('Unused input')
			continue
		#Array describing which values are when gradients are both positive or both negative
		simW = T.neq((T.eq((pw > 0), (gradients[-1] > 0))), (T.eq((pw < 0), (gradients[-1] <
			0))))

		#Array describing which values are when gradients are in opposite directions
		diffW = ((pw > 0) ^ (gradients[-1] > 0)) * (T.neq(pw, 0) * T.neq(gradients[-1], 0))
		updates.append((p, p - (T.sgn(gradients[-1]) * dw * (T.eq(diffW, 0)))))
		updates.append((dw, T.switch(diffW, dw *
			0.5, T.switch(simW, dw * 1.2, dw))))
		updates.append((pw, (T.sgn(gradients[-1]) * dw * (T.eq(diffW, 0)))))

	storage = prevw + deltaw
	print("\nDone with updates")

	return (storage, updates)

class AutoEncoder:
	def __init__(self, *dim, **kwargs):
		momentum = kwargs.get('momentum', 0)
		alpha = kwargs.get('alpha', 0.01)
		rprop = kwargs.get('rprop', False)
		init_size = kwargs.get('init_size', 1)
		verbose = kwargs.get('verbose', False)
		in_type = kwargs.get('in_type', 'linear')
		layers = []

		self.x = kwargs.get('in_var', T.matrix('Generalized Input'))
		layers.append(Layer(dim[0], dim[1], in_var=self.x, init_size=init_size))
		for i in range(1, len(dim) - 1):
			layers.append(Layer(dim[i], dim[i+1], in_var=layers[-1].out, init_size=init_size))

		self.out = layers[-1].out
		decoder = []

		if len(dim) > 2:
			layer_type = 'sigmoid'
		else:
			layer_type = in_type
		decoder.append(Layer(dim[-1], dim[-2], in_var=self.out, init_size=init_size,
			layer_type=layer_type))
		for i in range(2, len(dim)):
			if i == len(dim) - 1:
				layer_type = in_type
			decoder.append(Layer(dim[-i], dim[-i-1], in_var=decoder[-1].out,
				init_size=init_size, layer_type=layer_type))


		self.reconstructed = decoder[-1].out

		self.encode = layers
		self.decode = decoder

		if verbose: print(len(layers), len(decoder))
		self.layers = self.encode + self.decode

		self.params =[]

		for l in self.layers:
			self.params = self.params + l.params


		print('Finished initialization')

	def learn(self,x, y, mode='vanilla', iterations=10):
		def getEncDecPair(layer):
			encoder = self.encode[layer]
			decoder = self.decode[-(layer + 1)]
			return (encoder, decoder)

		def getLearningFunction(layer, x, y):
			encoder, decoder = getEndDecPair(layer)

			newOut = encoder.getOutput(x)
			out = decoder.getOutput(newOut)

			mse = T.mean(T.sqr(out - y))

			return (x, mse)

		for i in range(self.encode):
			for j in range(iterations):
				pass

		#Warning: Not done at all

class ConvolutionLayer:
	def __init__(self, shape, in_var=T.matrix('input'), nonlinearity=T.nnet.sigmoid, init_size=0.1,
			deconv=False):
		if nonlinearity == None:
			def na(x):
				return x
			nonlinearty = na
		x = in_var
		self.x = x
		filt = theano.shared(value=(np.random.uniform(low=-1.0, high=1.0, size=shape) * init_size)).astype(theano.config.floatX)

		#If bias needs to be applied to every hidden unit, it should be 3d
		bias = theano.shared(value=(np.random.rand(shape[1]) * 0.5 - 1) *
				init_size).astype(theano.config.floatX)
		if deconv:
			z = T.nnet.conv2d(x, filt, border_mode='full', subsample = (1,1))
		else:
			z = T.nnet.conv2d(x, filt, border_mode='valid', subsample=(1,1))

		self.out = nonlinearity(z + bias.dimshuffle('x', 0, 'x', 'x'))
		self.w = filt
		self.b = bias
		self.params = [filt, bias]

class ConvolutionalAutoEncoder:
	'''
	Reminder: The goal of this is purely to make dreams, nothing else
	'''
	def __init__(self, *dim, **kwargs):

		x = T.tensor4('input')
		self.x = x
		init_size = kwargs.get('init_size', 0.1)

		encode = []
		encode.append(ConvolutionLayer(dim[0], in_var=x, init_size=init_size, deconv=False))
		for i in range(1, len(dim)):
			encode.append(ConvolutionLayer(dim[i], in_var=layers[-1].out, init_size=init_size, deconv=False))

		decoder = []
		self.out = encode[-1].out

		def swapElements(dimension):
			d = list(dimension)
			d[0], d[1] = d[1], d[0]
			return tuple(d)

		decoder.append(ConvolutionLayer(swapElements(dim[-1]), in_var=encode[-1].out, init_size=init_size,
			deconv=True))
		#gotta implement the decoding step lol
		for i in range(2, len(dim)):
			nonlinearity = T.nnet.sigmoid
			if i == len(dim) - 1:
				nonlinearity = None
			decoder.append(ConvolutionLayer(swapElements(dim[-i]), in_var=decoder[-1].out, init_size=init_size,
				deconv=True, nonlinearity=nonlinearity))

		print(decoder[-1].w.get_value().shape)
		self.reconstructed = decoder[-1].out

		layers = encode + decoder

		self.params = []
		for l in layers:
			self.params += l.params


class FFClassifier:
	def __init__(self, *dim, **kwargs):

		x = T.matrix('input')
		self.x = x
		init_size = kwargs.get('init_size', 0.1)

		layers = []
		layers.append(Layer(dim[0], dim[1], in_var=x, init_size=init_size))
		for i in range(1, len(dim) - 1):
			layers.append(Layer(dim[i], dim[i+1], in_var=layers[-1].out, init_size=init_size))

		self.out = layers[-1].out
		self.layers = layers

		self.params = []
		for l in layers:
			self.params = self.params + l.params

def miniBatchLearning(x, y, batchSize, updateFunction, verbose=False, epochs=1):
	train_error = []
	np.set_printoptions(precision=10)
	if batchSize <= 0:
		batchSize = x.shape[0]
	for j in range(epochs):
		for i in range(0, x.shape[0], batchSize):
			error = updateFunction(x[i:(i+batchSize)], y[i:(i+batchSize)])
			train_error.append(error)
			if verbose: print('{0:10} Epoch: {1:4}'.format(np.round(error, 10), round((j +
				(i/x.shape[0])), 2)))
	return train_error

def AETester():
	images, labels = readMNISTData(6000)
	xcv, ycv = readcv(100)
	ae = AutoEncoder(784, 600, init_size=10)


#	images = images / images.max()

	genImage = theano.function([ae.x], ae.reconstructed)

	y = T.matrix('correct output')
	yl = T.matrix('Correct labels')
	mse = T.mean(T.sqr(y - ae.reconstructed))

	crossEntrop = -T.mean(yl * T.log(ae.out) + (1 - yl) * T.log(1 - ae.out))

	(adamStorage, adam) = generateAdam(ae.params, mse, alpha=1)

	learn = theano.function([ae.x, y], mse, updates=adam)
	train_error = miniBatchLearning(images, images, 100, learn, verbose=True, epochs=10)

#	pickle.dump(ae, open('autoencoder.pkl', 'wb'))

	plt.plot(np.arange(len(train_error)), train_error)
	plt.show()

	for i in range(len(images)):
		generated_image =genImage(images[i].reshape(1, 784))
		print("Min: {0:10} Max: {1:10}".format(generated_image.min(), generated_image.max()))
		plt.imshow(generated_image.reshape(28, 28), cmap='Greys', interpolation='none')
		plt.figure()
		plt.imshow(images[i].reshape(28, 28), cmap='Greys', interpolation='none')
		plt.show()

def NNTester():
	images, labels = readMNISTData(6000)
	xcv, ycv = readcv(1000)

	y = T.matrix('Correct Labels')
	nn = FFClassifier(784, 1000, 700, 500, 300, 10)
	error = -T.mean(y * T.log(nn.out) + (1-y) * T.log(1- nn.out))

	dupdates = generateVanillaUpdates(nn.params, error, alpha=0.01)
	(s, adam) = generateAdam(nn.params, error, alpha=0.001)
	(st, rprop) = generateRpropUpdates(nn.params, error, 0.1)
	(sto, rms) = generateRmsProp(nn.params, error, alpha=0.001, decay=0.9)
	(stor, adadelta) = generateAdadelta(nn.params, error, alpha=1, decay=0.9)
	(stora, adagrad) = generateAdagrad(nn.params, error, alpha=0.01)

	updateRules = [(dupdates, "SGD"), (adam, "Adam"), (rms, "RMS"), (adadelta, "Adadelta"), (adagrad, "Adagrad")]

	figure, ax = plt.subplots()
	for u in updateRules:
		reset(nn.params)
		learn = theano.function([nn.x, y], error, updates=u[0], allow_input_downcast=True)
		start_time = time.perf_counter()
		train_error = miniBatchLearning(images, labels, 100, learn, verbose=False, epochs=10)
		print(u[1], " took ", (time.perf_counter() - start_time))
		ax.plot(np.arange(len(train_error)), train_error, label=u[1])

	plt.yscale('log', nonposy='clip')
	ax.legend(loc='upper right')
	plt.xlabel("Iterations")
	plt.ylabel("MSE")
	figure.suptitle("Learning Style Comparison")
	plt.savefig("test.png")
	plt.show()


def ConvolutionDreamerTest():
	conv = ConvolutionalAutoEncoder((2, 3, 15, 15))

	i = 0

	'''
	def googleImageDownloader(start=0):
		from apiclient.discovery import build
		nonlocal i

		service = build("customsearch", "v1",
			   developerKey="AIzaSyBkd2lwAWzEKWjRbB2rlRM5OU_OBvz7u5w")

		res = service.cse().list(
			q='flower',
			cx='015897475084883013797:9dpu7012l6o',
			searchType='image',
			num=10,
			start=start,
			imgType='photo',
			fileType='png',
			imgSize='huge',
			safe= 'off').execute()

		import urllib.request as urllib
		def downloadImage(url, i):
			urllib.urlretrieve(url, 'imageDataSet/' + str(i))

		if not 'items' in res:
			print ('No result !!\nres is: {}'.format(res))
		else:
			for item in res['items']:
				print('{}:\n\t{}'.format(item['title'], item['link']))
				i += 1
				try:
					downloadImage(item['link'], i)
				except:
					print('Failed to download:', item['title'])
	'''
	from PIL import Image
	def convertImageToArray(index):
		filename = 'imageDataSet/' + str(index)
		im = Image.open(filename)
		im.load()
		return np.asarray(im)

	images = []
	for i in range(1, 2):
		try:
			images.append(convertImageToArray(i)[:, :, :3])
		except FileNotFoundError:
			pass

	plt.imshow(images[0])
	print(images[0].shape)
	plt.show()
	reconstruct = theano.function([conv.x], conv.out)
	reconstruct(images[0].reshape(1, images[0].shape[-1], images[0].shape[0], images[0].shape[1]))

if __name__ == '__main__':
	AETester()
