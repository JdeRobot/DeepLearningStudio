from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, BatchNormalization, Dropout, ConvLSTM2D, Reshape, \
    Activation, MaxPooling2D
from tensorflow.keras.optimizers import Adam


def pilotnet_x3_time_distributed(img_shape, learning_rate):
    model = Sequential()
    model.add(BatchNormalization(epsilon=0.001, axis=-1, input_shape=img_shape))
    model.add(TimeDistributed(Conv2D(24, (5, 5), strides=(2, 2), activation="relu", padding='same')))
    model.add(TimeDistributed(Conv2D(36, (5, 5), strides=(2, 2), activation="relu", padding='same')))
    model.add(TimeDistributed(Conv2D(48, (5, 5), strides=(2, 2), activation="relu", padding='same')))
    model.add(TimeDistributed(Conv2D(64, (3, 3), strides=(1, 1), activation="relu", padding='same')))
    model.add(TimeDistributed(Conv2D(64, (3, 3), strides=(1, 1), activation="relu", padding='same')))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(32, return_sequences=True))
    model.add(LSTM(32, return_sequences=True))
    model.add(LSTM(32, return_sequences=True))
    model.add(LSTM(32))
    model.add(Dense(3))
    adam = Adam(learning_rate=learning_rate)
    model.compile(optimizer=adam, loss="mse", metrics=['mse', 'mae'])
    return model
