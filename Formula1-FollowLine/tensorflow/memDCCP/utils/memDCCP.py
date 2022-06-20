from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, TimeDistributed, Dense, LSTM, Conv3D, BatchNormalization, ConvLSTM2D, \
    Dropout, Reshape, Activation, MaxPooling3D, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model


# memDCCP
def memDCCP(img_shape, learning_rate):
    model = Sequential()
    model.add(BatchNormalization(epsilon=0.001, axis=-1, input_shape=img_shape))

    model.add(Conv3D(24, (5, 5, 5), strides=(2, 2, 2), activation="relu", padding='same'))
    model.add(Conv3D(36, (5, 5, 5), strides=(2, 2, 2), activation="relu", padding='same'))
    model.add(Conv3D(48, (5, 5, 5), strides=(2, 2, 2), activation="relu", padding='same'))
    model.add(Conv3D(64, (3, 3, 3), strides=(1, 1, 1), activation="relu", padding='same'))
    model.add(Conv3D(64, (3, 3, 3), strides=(1, 1, 1), activation="relu", padding='same'))

    model.add(Reshape((1, 13, 64, 7)))

    model.add(ConvLSTM2D(filters=8, kernel_size=(5, 5), padding="same", return_sequences=True))
    model.add(ConvLSTM2D(filters=8, kernel_size=(5, 5), padding="same", return_sequences=True))
    model.add(ConvLSTM2D(filters=8, kernel_size=(5, 5), padding="same", return_sequences=True))
    model.add(ConvLSTM2D(filters=8, kernel_size=(5, 5), padding="same", return_sequences=False))

    model.add(Flatten())

    model.add(Dense(50, activation="relu"))
    model.add(Dense(10, activation="relu"))
    model.add(Dense(2))

    adam = Adam(lr=learning_rate)
    model.compile(optimizer=adam, loss="mse", metrics=['mse', 'mae'])
    return model