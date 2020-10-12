from importlib import import_module

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

from fed_learning.server.global_model_config import GlobalModelConfig
from fed_learning.server.global_model import GlobalModel


# NOTE mlei: for development purposes only, as it aims for shorter testing cycles
#            reduced NN width and depth
from test.demo.util import resnet


def build_model(model_config: GlobalModelConfig, test_data):
    return CIFARResNet(model_config, test_data)

class CIFARResNet(GlobalModel):
    
    def __init__(self, model_config: GlobalModelConfig, test_data):
        super(CIFARResNet, self).__init__(model_config, test_data)

    @classmethod
    def build_model(cls):
        # ~less parameters
        do = 0.2
        model = keras.Sequential()

        model.add(Conv2D(filters=6, kernel_size=5, strides=1, kernel_initializer='he_normal', padding='valid', activation='relu', input_shape=(32, 32, 3)))
        model.add(MaxPooling2D())
        model.add(Conv2D(filters=16, kernel_size=5, strides=1, kernel_initializer='he_normal', padding='valid', activation='relu'))
        model.add(MaxPooling2D())

        model.add(Flatten())
        # model.add(layers.Dropout(do))
        model.add(Dense(units=120, kernel_initializer='he_normal', activation='relu'))
        # model.add(layers.Dropout(do))
        model.add(Dense(units=84, kernel_initializer='he_normal', activation='relu'))
        # model.add(layers.Dropout(do))
        model.add(Dense(units=10, activation='softmax'))

        # n = 3
        # depth = n * 6 + 2
        # input_shape = (32, 32, 3)
        # model = resnet.resnet_v1(input_shape, depth)
        return model

