import sys
import pickle
import codecs
import numpy as np

from fed_learning.client.aggregator import Aggregator
from fed_learning.client.local_model import LocalModel
from fed_learning.client.local_model_config import LocalModelConfig
from fed_learning.client.dataset import DataSet
from fed_learning.message import *
from fed_learning.crypto.crypto_interface import CryptoInterface

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def build_aggregator(sio, dataset: DataSet, *args) -> Aggregator:
    return SecureAggregator(sio, dataset, *args)

class SecureAggregator(Aggregator):

    def __init__(self, sio, dataset: DataSet):
        logger.info('Initializing client-side aggregator with secure aggregation')
        super().__init__(sio, dataset)
        self.ci = CryptoInterface()
        self.register_at_server()

    def generate_training_finished_msg(self, update, msg):
        logger.info('Generate Randproofs')
        blindings = msg.content['blindings']

        training_finished_msg = TrainingFinishedMsg(
            self.model.model_config.model_id,
            self.model.get_current_round_id(),
            content= self.create_randproof(update, blindings)
        )
        print("Sending traning finished msg")
        return training_finished_msg

    def create_randproof(self, update, blindings):
        # print(update[0:6], update.shape, update.dtype)
        # (randproofs, update_enc, rand_commits) = self.ci.create_randproof(update, blindings)
        update_enc = self.ci.commit(update, blindings)
        print("Done creating commits")
        #
        # update_dec = self.ci.extract_values(update_enc)
        # print("Extracted", np.sum(update_dec))

        return {
                # 'randproofs': randproofs,
                'update_enc': update_enc,
                # 'rand_commits': rand_commits
            }