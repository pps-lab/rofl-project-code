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
    return RangeAggregator(sio, dataset, *args)

class RangeAggregator(SecureAggregator):

    def __init__(self, sio, dataset: DataSet):
        super().__init__(sio, dataset)
        logger.info('Augmenting secure aggregator with rangeproofs')

    def generate_training_finished_msg(self, update, msg):
        logger.info('Clipping values to %d bit range' % self.crypto_config.value_range)
        update_clipped = self.ci.clip_to_range(update, self.crypto_config.value_range)
        training_finished_msg = super().generate_training_finished_msg(update_clipped, msg)
        
        blindings = msg.content['blindings']
        logger.info('Generating rangeproofs with %d bit range, %d partitions' % (self.crypto_config.value_range, self.crypto_config.n_partition))
        (rangeproofs, commits) = self.ci.create_rangeproof(
            update_clipped,
            blindings,
            self.crypto_config.value_range,
            self.crypto_config.n_partition)

        update_enc = training_finished_msg.content['update_enc']
        assert(self.ci.commits_equal(update_enc, commits))
        training_finished_msg.content['rangeproofs'] = rangeproofs
        return training_finished_msg
