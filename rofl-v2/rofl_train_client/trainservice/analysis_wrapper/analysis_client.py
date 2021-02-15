

import tensorflow as tf
import numpy as np

from src.client import Client
from src.tf_model import Model
import src.config as cnf
from src.data.tf_data import Dataset
from src.main import load_model

from . import util
from .data_loader import load_dataset

import logging

class AnalysisClientWrapper():

    def __init__(self, config_path):
        self.config = cnf.load_config(config_path)
        print(self.config)

        dataset = load_dataset(self.config)

        malicious = False
        self.client = Client("dummy_id", self.config.client, dataset, malicious)

        self.model = self.load_model(self.config)
        self.client.set_model(self.model)

    def set_weights(self, w):
        unflattened = util.unflatten(w, self.model.get_weights())
        self.client.set_weights(unflattened)

    def train(self, round):
        logging.info(f"Training round {round}")
        self.client.train(round)
        logging.info(f"Done training round {round}")
        return util.flatten_update(self.client.weights)

    def load_model(self, config):
        if config.environment.load_model is not None:
            model = tf.keras.models.load_model(config.environment.load_model) # Load with weights
        else:
            model = Model.create_model(
                config.client.model_name, config.server.intrinsic_dimension, config.client.model_weight_regularization)
        return model


