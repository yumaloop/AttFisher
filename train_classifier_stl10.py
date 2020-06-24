import numpy as np
import keras
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Activation, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from stl10_utils import read_all_images, read_labels

# this is the size of our encoded representations
num_epochs = 1000
batch_size = 50

x_train = read_all_images("./dataset/STL10/data/stl10_binary/train_X.bin")
x_test  = read_all_images("./dataset/STL10/data/stl10_binary/test_X.bin")
y_train = read_labels("./dataset/STL10/data/stl10_binary/train_y.bin")
y_test  = read_labels("./dataset/STL10/data/stl10_binary/test_y.bin")


# x_train (5000, 96, 96, 3)
# x_test (8000, 96, 96, 3)
# y_train (5000,)
# y_test (8000,)

x_train = x_train.astype('float32') / 255.
x_test  = x_test.astype('float32') / 255.
# Ohe-Hot encoding
y_train = keras.utils.np_utils.to_categorical(y_train-1, 10) 
y_test  = keras.utils.np_utils.to_categorical(y_test-1, 10) 

x_test = x_test[:100]
y_test = y_test[:100]


def build_cnn():
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

    x = Conv2D(128, (1, 1), activation='relu', padding='same')(x)
    x = Flatten()(x)
    x = Dense(10)(x)
    y = Activation('softmax')(x)

    # this model maps an input to its reconstructionS
    model = Model(input_x, y)
    return model

def build_vgg16():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(96, 96, 3))
    x = base_model.output
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    pred = Dense(10, activation='sigmoid')(x)
    model = Model(base_model.input, pred)
    
    for layer in base_model.layers:
        layer.trainable = False
    
    return model

model = build_vgg16()
model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy')

# Train autoencoder for 100 epochs
for epoch in range(int(num_epochs / 10)):
    model.fit(x_train, y_train, 
                epochs=10, batch_size=batch_size, shuffle=True, 
                validation_data=(x_test, y_test), verbose=1)
    
    filepath="./trained_model/vgg16_stl10_epoch{}.h5".format((epoch+1) * 10)
    model.save(filepath)
    
    print("Trial:", (epoch+1)*10, "has ended.")
