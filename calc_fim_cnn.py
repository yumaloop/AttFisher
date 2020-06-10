import numpy as np
import tensorflow as tf

import keras
import keras.backend as K

import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras.datasets import mnist

from tqdm import tqdm as tqdm
from scipy.stats import norm

with keras.utils.CustomObjectScope({'GlorotUniform': keras.initializers.glorot_uniform()}):
    model = keras.models.load_model('trained_model/trained_cnn_v0.h5', compile=False)
    model.summary()

(x_train, y_train), (x_test, y_test) = mnist.load_data()
del mnist

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
# Ohe-Hot encoding
y_train = keras.utils.np_utils.to_categorical(y_train) 
y_test  = keras.utils.np_utils.to_categorical(y_test) 


y_true = K.placeholder((None, 10))

# loss = K.variable()
loss = tf.keras.losses.MSE(y_true, model.output)
input_layer = model.layers[0].output
hessian = tf.hessians(loss, input_layer)[0]


hessian_matrix_list = []

N = len(x_test)
# N = 1000
for idx in tqdm(range(N)):
    x = x_test[idx].reshape(1, 28, 28)
    y = y_test[idx].reshape(1, -1)

    tf_session = K.get_session()
    hessian_matrix = hessian.eval({model.input: x, y_true: y}, session=tf_session)
    hessian_matrix = np.squeeze(hessian_matrix)
    hessian_matrix_list.append(hessian_matrix)
    del hessian_matrix
    
fisher_info_matrix= np.zeros((784,784))
for idx in tqdm(range(len(hessian_matrix_list))):
    fisher_info_matrix = (idx / (idx + 1)) * fisher_info_matrix + (1 / (idx + 1)) * hessian_matrix_list[idx]

np.save('data/fim_cnn_xtest.npy', fisher_info_matrix)
