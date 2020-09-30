import os
import sys
from os.path import dirname, abspath, join

module_path = dirname(abspath(__file__))
sys.path.insert(0, dirname(module_path))

import unittest
import time
import numpy as np
import argparse
from importlib import import_module
import keras

from fed_learning.client.client import Client
from fed_learning.client.dataset import DataSet

from test.config_loader import ConfigLoader
from test.data_loader import DataLoader
from test.file_logger import setup_file_logger

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ROOT = abspath(dirname(module_path))
DATA = join(ROOT, 'data')

class TestClient(object):

    def __init__(self,id, config: ConfigLoader):
        self.id = id
        self.config = config
        self.dataset_prefix = config.dataset + '_'

        setup_file_logger(logger, self.config, 'test_client_' + self.id + '.log')

        transform_handler = self._load_transformer()
        loader = DataLoader(DATA, self.dataset_prefix, self.id, transform_handler)
        self.dataset = DataSet(*loader.load_data())

        self.aggregator_build_handler = self._load_aggregator_builder()

        self.client = Client(config.host, config.port, self.id, self.dataset, aggregator_builder=self.aggregator_build_handler)

    def _load_transformer(self):
        transform_handler = None
        if self.config.transformer is not None:
            logger.info('Importing %s module', self.config.transformer)
            transformer_module = import_module(self.config.transformer)
            transform_handler = transformer_module.transform
        return transform_handler

    def _load_aggregator_builder(self):
        module_name = 'fed_learning.client.aggregator.' + self.config.aggregator
        logger.info('Importing aggregator from %s' % module_name)
        aggregator_module = import_module(module_name)
        return aggregator_module.build_aggregator


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Instantiate client with provided data set')
    parser.add_argument('id', type=str, help="Client id")
    parser.add_argument('config_file_path', type=str, help="Configuration file location")
    args = parser.parse_args()

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s in %(module)s: %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    config = ConfigLoader(args.config_file_path)
    test_client = TestClient(args.id, config)
