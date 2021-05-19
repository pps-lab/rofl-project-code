

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

    def __init__(self, config_path, dataset_path):
        self.config = cnf.load_config(config_path)
        print(self.config)

        batch_size = self.config.client.benign_training.batch_size
        augment_data = self.config.dataset.augment_data
        self.dataset = load_dataset(dataset_path, batch_size, augment_data)

        malicious = False
        self.client = Client("dummy_id", self.config.client, self.dataset, malicious)

        self.model = self.load_model(self.config)

        self.num_weights = len(util.flatten_update(self.model.get_weights()))

    def set_weights(self, w):
        assert self.num_weights == len(w), f"The number of parameters in the crypto configuration ({len(w)}) " \
                                           f"must match the number of parameters in the model ({self.num_weights})!"
        unflattened = util.unflatten(w, self.model.get_weights())
        self.client.set_weights(unflattened)

    def train(self, round):
        logging.info(f"Training round {round}")
        self.model.set_weights(self.client.weights)
        # self.evaluate()
        self.client.set_model(self.model)
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

    def evaluate(self):
        self.model.compile(tf.keras.optimizers.SGD(), # Dummy, as we are not training
                           tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                           metrics=['acc'])
        score = self.model.evaluate(self.dataset.x_test, self.dataset.y_test, verbose=1)
        print(f"Score {score}")

