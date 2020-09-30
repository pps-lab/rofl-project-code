
import tensorflow.keras as keras
from keras.regularizers import l2
from tensorflow.keras import layers


def build_modelc(l2_reg):

    do = 0.2
    model = keras.Sequential()

    # model.add(layers.Dropout(0.2, noise_shape=(32, 32, 3)))
    model.add(layers.Conv2D(filters=96, kernel_size=3, strides=1, kernel_initializer='he_normal', padding='same', activation='relu', kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg), input_shape=(32, 32, 3)))
    model.add(layers.Conv2D(filters=96, kernel_size=3, strides=1, kernel_initializer='he_normal', padding='same', activation='relu', kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg)))
    model.add(layers.Conv2D(filters=96, kernel_size=3, strides=2, kernel_initializer='he_normal', padding='same', activation='relu', kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg)))
    # model.add(layers.Dropout(0.5))

    model.add(layers.Conv2D(filters=192, kernel_size=3, strides=1, kernel_initializer='he_normal', padding='same', activation='relu', kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg)))
    model.add(layers.Conv2D(filters=192, kernel_size=3, strides=1, kernel_initializer='he_normal', padding='same', activation='relu', kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg)))
    model.add(layers.Conv2D(filters=192, kernel_size=3, strides=2, kernel_initializer='he_normal', padding='same', activation='relu', kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg)))
    # model.add(layers.Dropout(0.5))

    model.add(layers.Conv2D(filters=192, kernel_size=3, strides=1, kernel_initializer='he_normal', padding='same', activation='relu', kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg)))
    model.add(layers.Conv2D(filters=192, kernel_size=1, strides=1, kernel_initializer='he_normal', padding='valid', activation='relu', kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg)))
    model.add(layers.Conv2D(filters=10, kernel_size=1, strides=1, kernel_initializer='he_normal', padding='valid', activation='relu', kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg)))

    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(units=10, activation='softmax'))

    return model