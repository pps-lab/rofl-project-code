

import os
import sys
module_loc = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,os.path.dirname(module_loc))

# import fed_learning.util.async_tools

import unittest
import logging 
import time
import argparse
from importlib import import_module
import keras

from fed_learning.server.server import Server
from fed_learning.server.global_model import GlobalModel
from fed_learning.server.global_model_config import GlobalModelConfig
from fed_learning.server.server_crypto_config import ServerCryptoConfig

from test.config_loader import ConfigLoader
from test.file_logger import setup_file_logger

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class TestServer(object):

    def __init__(self, config: ConfigLoader):
        self.config = config

        setup_file_logger(logger, self.config, 'test_server.log')

        self.test_dataset = self._load_and_transform_test_data()
        self.model_config = self._generate_model_config()
        self.model = self._load_model(self.model_config)
        self.crypto_config = self._generate_crypto_config()

        self.aggregator_build_handler = self._load_aggregator_builder()

        self.server = Server(self.config.host, self.config.port, self.model, self.crypto_config, aggregator_builder=self.aggregator_build_handler)
        self.server.start()

    def _load_and_transform_test_data(self):
        logger.info('Importing test data from keras.datasets.%s', self.config.dataset)
        data_module = import_module('keras.datasets.' + self.config.dataset)
        (x_train, y_train), (x_test, y_test) = data_module.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0

        if self.config.transformer is not None:
            logger.info('Importing %s module', self.config.transformer)
            transformer_module = import_module(self.config.transformer)
            transform_handler = transformer_module.transform
            _, _, x_test, y_test = transform_handler(x_train,
                                                     y_train,
                                                     x_test,
                                                     y_test)
        return x_test, y_test

    def _generate_model_config(self):
        return GlobalModelConfig(self.config.iterations,
                                 self.config.num_clients,
                                 self.config.client_batch_size,
                                 self.config.num_local_epochs,
                                 self.config.optimizer,
                                 self.config.learning_rate,
                                 self.config.loss,
                                 self.config.metrics,
                                 self.config.image_augmentation,
                                 self.config.lr_decay,
                                 self.config.probabilistic_quantization,
                                 self.config.fp_bits,
                                 self.config.fp_frac,
                                 self.config.value_range)

    def _load_model(self, model_config: GlobalModelConfig):
        model_module = import_module(config.model)
        return model_module.build_model(model_config, self.test_dataset)

    def _load_aggregator_builder(self):
        module_name = 'fed_learning.server.aggregator.' + self.config.aggregator
        logger.info('Importing aggregator from %s' % module_name)
        aggregator_module = import_module(module_name)
        return aggregator_module.build_aggregator

    def _generate_crypto_config(self):
        return ServerCryptoConfig(self.config.fp_bits,
                                  self.config.fp_frac,
                                  self.config.value_range,
                                  self.config.n_partition,
                                  self.config.l2_value_range,
                                  self.config.optimistic_starting)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Instantiate server for federated learning from config file")
    parser.add_argument('config_file_path', type=str, help="Configuration file location")
    args = parser.parse_args()

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s in %(module)s: %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    config = ConfigLoader(args.config_file_path)
    test_server = TestServer(config)