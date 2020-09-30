import keras
import numpy as np

def transform(x_train, y_train, x_test, y_test):

    x_tr = np.array(x_train, dtype=np.float32)
    y_tr = keras.utils.to_categorical(y_train, 10)
    x_te = np.array(x_test, dtype=np.float32)
    y_te = keras.utils.to_categorical(y_test, 10)

    return x_tr, y_tr, x_te, y_te