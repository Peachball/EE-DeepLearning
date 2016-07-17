from keras.models import Sequential
from keras.layers.recurrent import LSTM, GRU
import keras

config = {
        'model_file' : 'musicgen.h5',
        'freq_channels' : 512
        }

def load_model():
    model = Sequential()
    model.add(LSTM(800, input_dim=config['freq_channels']*2))
    pass

def generate_model():
    pass

def music_generator():
    pass

def learn_model():
    try:
        model = load_model()
    except:
        model = generate_model()
    X_dat, Y_dat = load_data()

    model.compile()
    model.fit(X_dat, Y_dat)

if __name__ == '__main__':
    learn_model()
