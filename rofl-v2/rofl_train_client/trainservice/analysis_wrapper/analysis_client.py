

import tensorflow as tf
import numpy as np

from src.client import Client
from src.tf_model import Model
import src.config as cnf
from src.data.tf_data import Dataset
from src.main import load_model
from .data_loader import load_dataset

class AnalysisClientWrapper():

    def __init__(self, config_path):
        self.config = cnf.load_config(config_path)
        print(self.config)

        dataset = load_dataset(self.config)
        print(dataset)
        malicious = False
        self.client = Client("dummy_id", self.config.client, dataset, malicious)

        self.model = self.load_model(self.config)
        self.client.set_model(self.model)

    def set_weights(self, w):
        unflattened = self._unflatten(w)
        self.client.set_weights(unflattened)

    def train(self, round):
        self.client.train(round)
        return self._flatten_update(self.client.weights)

    def _unflatten(self, w):
        weights = self.model.get_weights()
        sizes = [x.size for x in weights]
        split_idx = np.cumsum(sizes)
        update_ravelled = np.split(w, split_idx)[:-1]
        shapes = [x.shape for x in weights]
        update_list = [np.reshape(u, s) for s, u in zip (shapes, update_ravelled)]
        return update_list

    def _flatten_update(self, update):
        return np.concatenate([x.ravel() for x in update])


    def load_model(self, config):
        if config.environment.load_model is not None:
            model = tf.keras.models.load_model(config.environment.load_model) # Load with weights
        else:
            model = Model.create_model(
                config.client.model_name, config.server.intrinsic_dimension, config.client.model_weight_regularization)
        return model


