import json
import pickle
import codecs
import uuid
import numpy as np
import time

from keras import Model
from keras import backend as K


from fed_learning.server.global_model_config import GlobalModelConfig
from fed_learning.client.local_model_config import LocalModelConfig

import logging

from fed_learning.util import results_writer

logging.basicConfig(filename="server.log",
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# logger.

# NOTE mlei: list of optimizer strings
# all_classes = {
#     'sgd': SGD,
#     'rmsprop': RMSprop,
#     'adagrad': Adagrad,
#     'adadelta': Adadelta,
#     'adam': Adam,
#     'adamax': Adamax,
#     'nadam': Nadam,
#     'tfoptimizer': TFOptimizer,
# }

class GlobalModel(object):

    def __init__(self, model_config: GlobalModelConfig, test_data):

        self.model_config = model_config
        self.model = self.build_model()
        self.round_ids = []
        self.round_start_time = 0
        self.x_test, self.y_test = test_data
        logger.info('Initializing model with with dtype %s' % K.floatx())
        self.model.compile(loss=self.model_config.loss,
                optimizer=self.model_config.optimizer,
                metrics=self.model_config.metrics)
        logger.info('Number of parameters: %s' % self.model.count_params())
        self.model.summary(print_fn=logger.info)

    def evaluate(self):
        print(self.x_test[0])
        score = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        logger.info(
            '[EVAL] Test (round,loss,accuracy): (%d, %f, %f)' %
            (self.get_round_nr(), score[0], score[1]))

        end = time.time()
        results_writer.append(self.get_round_nr(), score[0], score[1], end - self.round_start_time)
        return score

    def build_model(self, seed=None) -> Model:
        raise NotImplementedError()

    def start_training_round(self):
        self.round_start_time = time.time()
        self.round_ids.append(str(uuid.uuid4))

    def get_current_round_id(self):
        return self.round_ids[-1]

    def get_round_nr(self):
        return len(self.round_ids)

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def get_weights(self):
        return self.model.get_weights()

    def get_all_weights(self):
        """Get all trainable weights. Optional"""
        return None

    def get_weights_seed(self):
        """Return seed to initialize weights with. Optional"""
        return None

    def params_count(self):
        return self.model.count_params()

    def serialize_model(self):
        return self.model.to_json()

    def serialize_model_name(self):
        """Implement if we wont transfer the model but the name"""
        return None

    def update_weights(self, update_flat):
        update = self._unflatten_update(update_flat)
        old_weights = self.model.get_weights()
        new_weights = []
        for w, u in zip(old_weights, update):
            new_weights.append(w + u)
        self.model.set_weights(new_weights)

    def _unflatten_update(self, update):
        weights = self.model.get_weights()
        sizes = [x.size for x in weights]
        split_idx = np.cumsum(sizes)
        update_ravelled = np.split(update, split_idx)[:-1]
        shapes = [x.shape for x in weights]
        update_list = [np.reshape(u, s) for s, u in zip (shapes, update_ravelled)]
        return update_list

