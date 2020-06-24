import numpy as np
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from stl10_utils import read_all_images

# this is the size of our encoded representations
num_epochs = 1000
batch_size = 50

# x_unlabeled = read_all_images(path_to_data="./dataset/STL10/data/stl10_binary/unlabeled_X.bin")
# x_unlabeled = x_unlabeled.astype('float32') / 255.
# x_unlabeled: (100000, 96, 96, 3)
x_train = read_all_images(path_to_data="./dataset/STL10/data/stl10_binary/train_X.bin")
x_train = x_train.astype('float32') / 255.
# x_train: (5000, 96, 96, 3)

def build_autoencoder():
    input_x = Input(shape=(96, 96, 3))
    x = Conv2D(32, (4, 4), activation='relu', padding='same')(input_x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (4, 4), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(128, (4, 4), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(128, (4, 4), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(128, (4, 4), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(128, (4, 4), activation='relu', padding='same')(x)
    x = MaxPooling2D((3, 3), padding='same')(x)

    z = Conv2D(128, (1, 1), activation='relu', padding='same')(x)

    x = UpSampling2D((3, 3))(z)
    x = Conv2D(128, (4, 4), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(128, (4, 4), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(128, (4, 4), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (4, 4), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (4, 4), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (4, 4), activation='sigmoid', padding='same')(x)

    # this model maps an input to its reconstructionS
    autoencoder = Model(input_x, decoded)
    return autoencoder


# configure model to use a per-pixel binary crossentropy loss, and the Adadelta optimizer
autoencoder = build_autoencoder()
autoencoder.summary()
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# Train autoencoder for 100 epochs
for epoch in range(int(num_epochs / 100)):
    autoencoder.fit(x_train, x_train, 
                    epochs=100, batch_size=batch_size, shuffle=True, 
                    validation_data=None, verbose=1)
    
    filepath="./trained_model/vae_stl10_epoch{}.h5".format((epoch+1) * 10)
    autoencoder.save(filepath)
    
    print("Trial:", (epoch+1)*100, "has ended.")
