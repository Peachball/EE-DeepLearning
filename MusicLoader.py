import numpy as np
import scipy.io.wavfile as wavUtil

def convertMusicFile(index, inputsize=100):
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
    print(data.shape)
    return data

if __name__ == '__main__':
    convertMusicFile(0)
