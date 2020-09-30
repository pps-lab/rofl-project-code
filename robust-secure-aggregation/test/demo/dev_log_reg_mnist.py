from importlib import import_module

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

from fed_learning.server.global_model_config import GlobalModelConfig
from fed_learning.server.global_model import GlobalModel

def build_model(model_config: GlobalModelConfig, test_data):
    return DevLogRegMNIST(model_config, test_data)

class DevLogRegMNIST(GlobalModel):
    
    def __init__(self, model_config: GlobalModelConfig, test_data):
        self.input_dim = 28*28
        self.output_dim = 10
        super(DevLogRegMNIST, self).__init__(model_config, test_data)

    @classmethod
    def build_model(cls):
        model = Sequential() 
        model.add(Dense(28 * 28, input_dim=10, activation='softmax'))
        return model
