import keras
import numpy as np

def transform(x_train, y_train, x_test, y_test):
    x_train_shape = (*np.shape(x_train), 1)
    x_test_shape = (*np.shape(x_test), 1)

    x_tr = np.reshape(x_train, x_train_shape)
    y_tr = keras.utils.to_categorical(y_train, 10)
    x_te = np.reshape(x_test, x_test_shape)
    y_te = keras.utils.to_categorical(y_test, 10)

    return x_tr, y_tr, x_te, y_te