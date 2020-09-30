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

    def test_try_set_weights(self):
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

        res_1 = self.global_model.evaluate()

        self.model_id = 42
        self.local_model_config = LocalModelConfig.from_config(self.config, self.model_id)
        self.local_model = LocalModel(self.local_model_config, data)

        weights = self.global_model.model.get_all_weights()

        tf.reset_default_graph()
        # with keras.backend.get_session() as sess:
        # self.global_model = build_model(self.global_model_config, (x_test, y_test))
        # self.global_model.model.set_all_weights(weights)
        json_dense = '{"class_name": "ExtendedModel", "config": {"name": "extendedmodel_1", "layers": [{"name": "input_1", "class_name": "InputLayer", "config": {"batch_input_shape": [null, 28, 28, 1], "dtype": "float32", "sparse": false, "name": "input_1"}, "inbound_nodes": []}, {"name": "r_proj_conv2d_1", "class_name": "RProjConv2D", "config": {"offset_creator_id": "dense", "filters": 8, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"name": "r_proj_conv2d_2", "class_name": "RProjConv2D", "config": {"offset_creator_id": "dense", "filters": 4, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["r_proj_conv2d_1", 0, 0, {}]]]}, {"name": "max_pooling2d_1", "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "inbound_nodes": [[["r_proj_conv2d_2", 0, 0, {}]]]}, {"name": "flatten_1", "class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "inbound_nodes": [[["max_pooling2d_1", 0, 0, {}]]]}, {"name": "r_proj_dense_1", "class_name": "RProjDense", "config": {"name": "r_proj_dense_1", "trainable": true, "dtype": "float32", "offset_creator_id": "dense", "units": 32}, "inbound_nodes": [[["flatten_1", 0, 0, {}]]]}, {"name": "r_proj_dense_2", "class_name": "RProjDense", "config": {"name": "r_proj_dense_2", "trainable": true, "dtype": "float32", "offset_creator_id": "dense", "units": 10}, "inbound_nodes": [[["r_proj_dense_1", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["r_proj_dense_2", 0, 0]], "vsize": 200}, "keras_version": "2.3.1", "backend": "tensorflow"}'
        n = model_from_json(json_dense, {
            'ExtendedModel': ExtendedModel,
            'RProjConv2D': RProjConv2D,
            'RProjDense': RProjDense
        })
        n.set_all_weights(weights)

        self.global_model.model = n
        res_2 = self.global_model.evaluate()

        # self.local_model.set_model(n)
        #
        # self.local_model.setup_new_round(1, n.get_weights())
        # self.local_model.train_one_round()

        print(res_1, res_2)