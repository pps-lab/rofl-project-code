import keras
import numpy as np
from keras.utils import np_utils 

def transform(x_train, y_train, x_test, y_test):
    input_dim = x_train[0].flatten().shape[0]
    x_tr = np.reshape(x_train, (len(x_train), input_dim)).astype('float32') 
    x_te = np.reshape(x_test, (len(x_test), input_dim)).astype('float32')
    x_tr /= 255
    x_te /= 255



    y_tr = np_utils.to_categorical(y_train, 10) 
    y_te = np_utils.to_categorical(y_test, 10)


    return x_tr, y_tr, x_te, y_te
