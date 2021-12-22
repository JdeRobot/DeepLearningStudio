from tensorflow.keras import Model
from tensorflow.keras.layers import Flatten, Dense, Conv2D, BatchNormalization, Dropout, ConvLSTM2D, Reshape, \
    Activation, MaxPooling2D, LSTM, Input, concatenate
from tensorflow.keras.optimizers import Adam

# NARX PilotNet
def narx_pilotnet_model(img_shape, ann_shape = (2,), learning_rate = 0.001):
    img_input = Input(shape = img_shape)
    ann_input = Input(shape = ann_shape)
    
    y_img = BatchNormalization(epsilon=0.001, axis=-1) (img_input)
    y_img = Conv2D(24, (5, 5), strides=(2, 2), activation="relu",padding='same') (y_img)
    y_img = Conv2D(36, (5, 5), strides=(2, 2), activation="relu",padding='same') (y_img)
    y_img = Conv2D(48, (5, 5), strides=(2, 2), activation="relu",padding='same') (y_img)
    y_img = Conv2D(64, (3, 3), strides=(1, 1), activation="relu",padding='same') (y_img)
    y_img = Conv2D(64, (3, 3), strides=(1, 1), activation="relu",padding='same') (y_img)
    y_img = Flatten() (y_img)
    
    y = concatenate([y_img, ann_input])
    y = Dense(1164, activation="relu") (y)
    y = Dense(100, activation="relu") (y)
    y = Dense(50, activation="relu") (y)
    y = Dense(10, activation="relu") (y)
    y = Dense(2, activation = "sigmoid") (y)
    
    model = Model(inputs = [img_input, ann_input], outputs = y)
    adam = Adam(learning_rate=learning_rate)
    model.compile(optimizer=adam, loss="mse", metrics=['mse', 'mae'])
    
    return model