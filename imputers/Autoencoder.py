from keras import Sequential, Model
from keras.layers import Dense, Flatten

class Autoencoders(Model):
    def __init__(self, input_dim):
        super(Autoencoders, self).__init__()

        self.encoder = Sequential([
            Flatten(),
            Dense(input_dim + 7 * 1, activation='tanh'),
            Dense(input_dim + 7 * 2, activation='tanh'),
            Dense(input_dim + 7 * 3)
        ])
        self.decoder = Sequential([
            Dense(input_dim + 7 * 2, activation='tanh'),
            Dense(input_dim + 7 * 1, activation='relu'),
            Dense(input_dim + 7 * 0)
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded