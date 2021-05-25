import tensorflow as tf
import numpy as np
from src.tf_model import Model


def load_model(config):
    np.random.seed(config.environment.seed)
    tf.random.set_seed(config.environment.seed)
    if config.environment.load_model is not None:
        model = tf.keras.models.load_model(config.environment.load_model) # Load with weights
    else:
        model = Model.create_model(
            config.client.model_name, config.server.intrinsic_dimension, config.client.model_weight_regularization)
    return model