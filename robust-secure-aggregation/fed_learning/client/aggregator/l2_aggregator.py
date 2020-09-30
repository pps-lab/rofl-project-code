import sys
import pickle
import codecs
import numpy as np

from fed_learning.client.aggregator import Aggregator
from fed_learning.client.aggregator.range_aggregator import RangeAggregator
from fed_learning.client.local_model import LocalModel
from fed_learning.client.local_model_config import LocalModelConfig
from fed_learning.client.dataset import DataSet
from fed_learning.message import *
from fed_learning.crypto.crypto_interface import CryptoInterface

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def build_aggregator(sio, dataset: DataSet, *args) -> Aggregator:
    return L2Aggregator(sio, dataset, *args)

class L2Aggregator(RangeAggregator):

    def __init__(self, sio, dataset: DataSet):
        super().__init__(sio, dataset)
        logger.info('Augmenting secure aggregator with l2 rangeproofs')

    """We override the create_randproof to create the squared randproof (extension) + 
        the l2 range proof of the sum of squares."""
    def create_randproof(self, update, blindings):
        l2_range = self.crypto_config.l2_value_range
        n_part = self.crypto_config.n_partition
        blindings_sq = self.ci.create_random_blinding_vector(len(update))
        self.print_l2_norm(update)

        logger.info('Generating l2 range proof and square randproof with %d bit range, %d partitions' % (l2_range, n_part))

        (randproofs, update_enc, rand_commits, square_commits, rangeproof, square_sum) = self.ci.create_l2proof(update,
                                                                                        blindings,
                                                                                        blindings_sq,
                                                                                        l2_range,
                                                                                        n_part)
        return {
            'randproofs': randproofs,
            'update_enc': update_enc,
            'rand_commits': rand_commits,
            'square_commits': square_commits,
            'l2_rangeproof': rangeproof,
            'square_sum_commit': square_sum
        }

    def print_l2_norm(self, update):
        norm = np.linalg.norm(update)
        logger.info(f"Update norm: {norm}")