from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, BatchNormalization, Dropout, ConvLSTM2D, Reshape, Activation, MaxPooling2D, InputLayer, LayerNormalization, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LeakyReLU
import tensorflow as tf


def pilotnet_model(img_shape, learning_rate):
    '''
    Model of End to End Learning for Self-Driving Cars (NVIDIA)
    '''
    model = Sequential()
    model.add(BatchNormalization(epsilon=0.001, axis=-1, input_shape=img_shape))
    # model.add(InputLayer(input_shape=img_shape))
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation="relu",padding='valid'))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation="relu",padding='valid'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation="relu",padding='valid'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation="relu",padding='valid'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation="relu",padding='valid'))
    # model.add(LayerNormalization(axis=3 , center=True , scale=True))
    # model.add(Dropout(0.1))
    model.add(Flatten())
    model.add(Dense(1164, activation="relu"))
    model.add(Dropout(0.1))
    model.add(Dense(100, activation="relu"))
    model.add(Dropout(0.1))
    model.add(Dense(50, activation="relu"))
    model.add(Dropout(0.1))
    model.add(Dense(10, activation="relu"))
    model.add(Dropout(0.1))
    model.add(Dense(2))
    adam = Adam(learning_rate=learning_rate)
    model.compile(optimizer=adam, loss="mse", metrics=['mse', 'mae'])
    return model