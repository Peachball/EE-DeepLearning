
config = {
        'model_file' : 'musicgen.h5'
        }

def load_model():
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
