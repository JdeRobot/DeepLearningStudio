from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, BatchNormalization, Dropout, ConvLSTM2D, Reshape, Activation, MaxPooling2D
from tensorflow.keras.optimizers import Adam


def pilotnet_x3_conv3d(img_shape, learning_rate):
    model = Sequential()
    model.add(BatchNormalization(epsilon=0.001, axis=-1, input_shape=img_shape))

    model.add(Conv3D(24, (5, 5, 5), strides=(2, 2, 2), activation="relu", padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Conv3D(36, (5, 5, 5), strides=(2, 2, 2), activation="relu", padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Conv3D(48, (5, 5, 5), strides=(2, 2, 2), activation="relu", padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Conv3D(64, (3, 3, 3), strides=(1, 1, 1), activation="relu", padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Conv3D(64, (3, 3, 3), strides=(1, 1, 1), activation="relu", padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(GlobalAveragePooling3D())
    # model.add(Flatten()) # Mantiene el orden de las características, en comparación con la GlobalaveragePooling que no lo mantiene. En temas temporales puede ser mejor utilizar Flatten

    model.add(Dense(1164, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(100, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(50, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(10, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(3))

    adam = Adam(learning_rate=learning_rate)
    model.compile(optimizer=adam, loss="mse", metrics=['mse', 'mae'])
    return model