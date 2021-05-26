
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2


def build_lenet5(input_shape=(32, 32, 3), l2_reg=None):

    do = 0.2
    regularizer = l2(l2_reg) if l2_reg is not None else None

    model = keras.Sequential()

    model.add(layers.Conv2D(filters=6, kernel_size=5, strides=1, kernel_initializer='he_normal', padding='valid', activation='relu', kernel_regularizer=regularizer, bias_regularizer=regularizer, input_shape=input_shape))
    model.add(layers.MaxPooling2D())
    model.add(layers.Conv2D(filters=16, kernel_size=5, strides=1, kernel_initializer='he_normal', padding='valid', activation='relu', kernel_regularizer=regularizer, bias_regularizer=regularizer))
    model.add(layers.MaxPooling2D())

    model.add(layers.Flatten())
    model.add(layers.Dropout(do))
    model.add(layers.Dense(units=120, kernel_initializer='he_normal', kernel_regularizer=regularizer, bias_regularizer=regularizer, activation='relu'))
    model.add(layers.Dropout(do))
    model.add(layers.Dense(units=84, kernel_initializer='he_normal', kernel_regularizer=regularizer, bias_regularizer=regularizer, activation='relu'))
    model.add(layers.Dropout(do))
    model.add(layers.Dense(units=10, activation='softmax'))

    return model