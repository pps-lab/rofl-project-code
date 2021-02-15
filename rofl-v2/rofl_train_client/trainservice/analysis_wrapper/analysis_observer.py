from src.client import Client
from src.tf_model import Model
import src.config as cnf
from src.data.data_loader import load_global_dataset
from src.attack_dataset_config import AttackDatasetConfig
import tensorflow as tf
import numpy as np
import logging

from rofl_train_client.trainservice.analysis_wrapper import util


class AnalysisObserver:
    """

    Loads global evaluation set from analysis.

    Maybe add logging in the future?

    """

    def __init__(self, config_path):
        self.config = cnf.load_config(config_path)
        print(self.config)

        self.attack_dataset = AttackDatasetConfig(**self.config.client.malicious.backdoor) \
            if self.config.client.malicious is not None and self.config.client.malicious.backdoor is not None else None

        self.model = self.load_model(self.config)
        self.dataset = load_global_dataset(self.config, np.array([], dtype=np.bool), self.attack_dataset)

    def evaluate(self, model_params, round):
        self.model.set_weights(util.unflatten(model_params, self.model.get_weights()))
        score = self.model.evaluate(self.dataset.x_test, self.dataset.y_test, verbose=1)
        logging.info(
            '[EVAL] Test (round,loss,accuracy): (%d, %f, %f)' %
            (round, score[0], score[1]))

    def load_model(self, config):
        if config.environment.load_model is not None:
            model = tf.keras.models.load_model(config.environment.load_model) # Load with weights
        else:
            model = Model.create_model(
                config.client.model_name, config.server.intrinsic_dimension, config.client.model_weight_regularization)
        return model