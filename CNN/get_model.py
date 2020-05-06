from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Input, UpSampling2D
from tensorflow.keras.models import Sequential
from tensorflow.python.keras import Model


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


def autoencoder(input_shape, optimiser):
    input_img = Input(shape=input_shape)

    nn = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
    nn = MaxPooling2D((2, 2), padding='same')(nn)
    nn = Conv2D(32, (3, 3), activation='relu', padding='same')(nn)
    encoded = MaxPooling2D((2, 2), padding='same')(nn)

    nn = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
    nn = UpSampling2D((2, 2))(nn)
    nn = Conv2D(64, (3, 3), activation='relu', padding='same')(nn)
    nn = UpSampling2D((2, 2))(nn)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(nn)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer=optimiser, loss='binary_crossentropy')

    return autoencoder