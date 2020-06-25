import numpy as np
import tensorflow as tf

# this is the size of our encoded representations
num_epochs = 1000
batch_size = 100

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
x_test  = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test  = tf.keras.utils.to_categorical(y_test, 10)

def build_lenet5():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)))
    model.add(tf.keras.layers.AveragePooling2D())
    model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
    model.add(tf.keras.layers.AveragePooling2D())
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=120, activation='relu'))
    model.add(tf.keras.layers.Dense(units=84, activation='relu'))
    model.add(tf.keras.layers.Dense(units=10, activation = 'softmax'))
    return model

model = build_lenet5()
model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy')

# Train autoencoder for 100 epochs
for epoch in range(int(num_epochs / 100)):
    model.fit(x_train, y_train, 
                epochs=100, batch_size=batch_size, shuffle=True, 
                validation_data=(x_test, y_test), verbose=1)
    
    filepath="./trained_model/lenet5_mnist_epoch{}.h5".format((epoch+1) * 100)
    model.save(filepath)
    
    print("Trial:", (epoch+1)*100, "has ended.")
