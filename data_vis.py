import matplotlib.pyplot as plt

def display(x, index):
    plt.imshow(x.transpose(0, 2, 3, 1)[index] * 255)
    return

def view_error(f):
    f = open(f, 'r')
    err = []
    for i in f:
        err.append(float(i))
    f.close()
    return err

def view_lstm_stats(data_dir='rundata'):
    from os.path import join

def view_EEData(dirname='EEimages/EE data'):
    from os import listdir
    from os.path import join, isfile, splitext, basename
    def load_data(filename):
        import pickle
        f = open(filename, 'rb')
        error = pickle.load(f)
        return error

    legend = []
    files = [join(dirname, f) for f in listdir(dirname) if isfile(join(dirname, f))]
    files.sort()
    for f in files[:]:
        _, ext = splitext(f)
        if ext == '.txt':
            continue
        err = load_data(f)
        handle, = plt.plot(err)
        legend += [(handle, splitext(basename(f))[0])]

    plt.legend(list(list(zip(*legend))[0]), list(list(zip(*legend))[1]))

    plt.xlabel("Iteration")
    plt.ylabel("Mean Squared Error")
    plt.yscale('log')
    # plt.savefig("RNN.png")
    plt.show()

if __name__ == '__main__':
    view_EEData()
