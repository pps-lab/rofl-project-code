import os

import keras
import sys
from keras.engine.saving import model_from_json

from client.dataset import DataSet
from fed_learning.intrinsic_dimension.keras_ext.engine_training import ExtendedModel
from fed_learning.intrinsic_dimension.keras_ext.rproj_layers import RProjConv2D, RProjDense

module_loc = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,os.path.dirname(module_loc))

import unittest
import numpy as np
import tensorflow as tf

from keras.datasets.mnist import load_data

from test.config_loader import ConfigLoader

from fed_learning.server.global_model_config import GlobalModelConfig
from fed_learning.client.local_model_config import LocalModelConfig
from fed_learning.client.local_model import LocalModel
from test.demo.dev_fed_mnist_cnn_subspace import build_model
# from test.demo.fed_mnist_cnn import build_model

class TestModel(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestModel, self).__init__(*args, **kwargs)
        self.config = ConfigLoader()
        (x_train, y_train), (x_test, y_test) = load_data()
        x_train = np.expand_dims(x_train / 255.0, 3)
        x_test = np.expand_dims(x_test / 255.0, 3)
        y_train = np.expand_dims(y_train, 1)
        y_test = np.expand_dims(y_test, 1)
        y_train = keras.utils.to_categorical(y_train, num_classes=10)
        y_test = keras.utils.to_categorical(y_test, num_classes=10)

        data = DataSet(x_train, y_train, x_test, y_test)
        self.global_model_config = GlobalModelConfig.from_config(self.config)
        self.global_model = build_model(self.global_model_config, (x_test, y_test))
        
        self.model_id = 42
        self.local_model_config = LocalModelConfig.from_config(self.config, self.model_id)
        self.local_model = LocalModel(self.local_model_config, data)

    def test_model_flatten_unflatten(self):
        self.local_model.set_model(self.global_model.model)
        local_weights = self.local_model.get_current_weights()
        local_weights_flat = self.local_model._flatten_update(local_weights)
        global_weights = self.global_model._unflatten_update(local_weights_flat)

        self.assertEqual(len(local_weights), len(global_weights))

        for x, y in zip(local_weights, global_weights):
            self.assertEqual(x.shape, y.shape)
            assert(np.array_equal(x, y))

    def test_model_iteration(self):
        with keras.backend.get_session() as sess:
            self.local_model.set_model(self.global_model.model)

            self.local_model.setup_new_round(1, self.global_model.model.get_weights())
            self.local_model.train_one_round()
            path = os.path.join(module_loc, "test_build")
            print(path)
            file_writer = tf.summary.FileWriter(path, sess.graph)

            # print(f"Weights: {weights}")
        file_writer.close()

    def test_eval(self):
        res = self.global_model.evaluate()

        weights = self.global_model.model.get_all_weights()
        self.global_model.model.set_all_weights(weights)

        res_2 = self.global_model.evaluate()
        print(res, res_2)