import src.config as cnf
from src.data.data_loader import load_global_dataset
from src.attack_dataset_config import AttackDatasetConfig
import tensorflow as tf
import numpy as np

from . import util
from .model_loader import load_model

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

        self.model = load_model(self.config)
        self.dataset = load_global_dataset(self.config, np.array([], dtype=np.bool), self.attack_dataset)

        self.model.compile(tf.keras.optimizers.SGD(), # Dummy, as we are not training
                           tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                           metrics=['acc'])

    def evaluate(self, model_params, round):
        self.model.set_weights(util.unflatten(model_params, self.model.get_weights()))
        score = self.model.evaluate(self.dataset.x_test, self.dataset.y_test, verbose=1)
        return score