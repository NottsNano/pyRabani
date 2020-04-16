import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Conv2D, Conv2DTranspose, MaxPooling2D, Flatten, Dropout, \
    BatchNormalization, Reshape, LeakyReLU, Activation
from tensorflow.keras.models import Sequential, Model
from tensorflow.python.keras import Input


def get_model(name, input_shape, output_num, optimiser):
    if name is "VGG":
        model = _VGG(input_shape, output_num, optimiser)
    elif name is "orig_model":
        model = _orig_model(input_shape, output_num, optimiser)
    else:
        model = None

    return model


def _orig_model(input_shape, output_num, optimiser):
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_num, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimiser,
                  metrics=['accuracy'])

    return model


def _VGG(input_shape, output_num, optimiser):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_num, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimiser,
                  metrics=['accuracy'])

    return model


def autoencoder(input_shape, optimiser, filters=(32, 64), latentdim=16):
    """Modified from https://www.pyimagesearch.com/2020/03/02/anomaly-detection-with-keras-tensorflow-and-deep-learning/"""
    # Build the encoder
    inputs = Input(shape=input_shape)
    x = inputs
    for f in filters:  # CONV => RELU => BN
        x = Conv2D(f, (3, 3), strides=2, padding="same")(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization()(x)

    volumeSize = K.int_shape(x)
    x = Flatten()(x)
    latent = Dense(latentdim)(x)
    encoder = Model(inputs, latent, name="encoder")

    # Build the decoder
    latentInputs = Input(shape=(latentdim,))
    x = Dense(np.prod(volumeSize[1:]))(latentInputs)
    x = Reshape((volumeSize[1], volumeSize[2], volumeSize[3]))(x)
    for f in filters[::-1]:
        x = Conv2DTranspose(f, (3, 3), strides=2,
                            padding="same")(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization()(x)
    x = Conv2DTranspose(input_shape[-1], (3, 3), padding="same")(x)
    outputs = Activation("sigmoid")(x)

    decoder = Model(latentInputs, outputs, name="decoder")

    # Build the autoencoder
    autoencoder = Model(inputs, decoder(encoder(inputs)),
                        name="autoencoder")
    autoencoder.compile(loss="mse", optimizer=optimiser)

    return (encoder, decoder, autoencoder)