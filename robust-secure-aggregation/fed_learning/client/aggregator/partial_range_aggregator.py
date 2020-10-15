import itertools
import sys
import pickle
import codecs
import numpy as np

from fed_learning.client.aggregator import Aggregator
from fed_learning.client.aggregator.secure_aggregator import SecureAggregator
from fed_learning.client.local_model import LocalModel
from fed_learning.client.local_model_config import LocalModelConfig
from fed_learning.client.dataset import DataSet
from fed_learning.message import *
from fed_learning.crypto.crypto_interface import CryptoInterface

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def build_aggregator(sio, dataset: DataSet, *args) -> Aggregator:
    return PartialRangeAggregator(sio, dataset, *args)

class PartialRangeAggregator(SecureAggregator):

    def __init__(self, sio, dataset: DataSet):
        super().__init__(sio, dataset)
        logger.info('Augmenting secure aggregator with partial rangeproofs')

    def generate_training_finished_msg(self, update, msg):
        random_seed = 420
        logger.info('Clipping values to %d bit range' % self.crypto_config.value_range)
        update_clipped = self.ci.clip_to_range(update, self.crypto_config.value_range)
        total_commitments = update_clipped.shape[0]
        print(f"Total: {total_commitments}")


        np.random.seed(random_seed)
        select_num = 3652 # int(total_commitments * 0.005)
        selected_indices = np.random.choice(range(0, total_commitments), select_num, replace=False)

        training_finished_msg = super().generate_training_finished_msg(update_clipped, msg)
        
        blindings = msg.content['blindings']
        update_to_proof = update_clipped[selected_indices]

        selected_blindings = self.ci.select_blinding_values(blindings, selected_indices)

        logger.info('Generating rangeproofs with %d bit range, %d partitions' % (self.crypto_config.value_range, self.crypto_config.n_partition))
        (rangeproofs, commits) = self.ci.create_rangeproof(
            update_to_proof,
            selected_blindings,
            self.crypto_config.value_range,
            self.crypto_config.n_partition)

        training_finished_msg.content['rangeproofs'] = rangeproofs
        return training_finished_msg
