from keras.models import Sequential
from keras.optimizers import Adam



def build_model():
    model = Sequential()

    model.add(Dense(output_dim=64, input_dim=100))
    model.add(Activation("relu"))
    model.add(Dense(output_dim=10))
    model.add(Activation("softmax"))
    model.compile(loss='mean_squared_error', 
                  optimizer=Adam(), 
                  metrics=['accuracy'])
    return model

def main():
    seqs, labels = load_sequences_and_labels()
    one_hot_encode_sequences()
    model = build_model()
    

if __name__ == '__main__':
    main()
