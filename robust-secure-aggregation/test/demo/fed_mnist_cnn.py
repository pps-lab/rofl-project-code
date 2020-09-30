from importlib import import_module

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

from fed_learning.server.global_model_config import GlobalModelConfig
from fed_learning.server.global_model import GlobalModel

def build_model(model_config: GlobalModelConfig, test_data):
    return DevFedMNISTCNN(model_config, test_data)

class DevFedMNISTCNN(GlobalModel):
    
    def __init__(self, model_config: GlobalModelConfig, test_data):
        super(DevFedMNISTCNN, self).__init__(model_config, test_data)

    @classmethod
    def build_model(cls):
        # ~5MB worth of parameters
        model = Sequential()
        # model.add(Conv2D(32, kernel_size=(3, 3),
        #                  activation='relu',
        #                  input_shape=(28, 28, 1)))
        # model.add(Conv2D(64, (3, 3), activation='relu'))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.25))
        # model.add(Flatten())
        # model.add(Dense(128, activation='relu'))
        # model.add(Dropout(0.25))
        # model.add(Dense(10, activation='softmax'))
        model.add(Conv2D(16, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=(28, 28, 1)))
        model.add(Conv2D(8, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(48, activation='relu'))
        model.add(Dense(10, activation='softmax'))

        return model
