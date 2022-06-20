from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, BatchNormalization, Dropout, ConvLSTM2D, Reshape, \
    Activation, MaxPooling2D, LSTM, Input
from tensorflow.keras.optimizers import Adam


# DEEPEST LSTM tinypilotnet
def deepest_lstm_tinypilotnet_model(img_shape, learning_rate):
    model = Sequential()
    model.add(Conv2D(8, (3, 3), strides=(2, 2), input_shape=img_shape, activation="relu"))
    model.add(Conv2D(8, (3, 3), strides=(2, 2), activation="relu"))
    model.add(Conv2D(8, (3, 3), strides=(2, 2), activation="relu"))
    model.add(Dropout(0.2))
    model.add(Reshape((1, 5, 11, 8)))
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
