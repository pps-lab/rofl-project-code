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

def build_aggregator(sio , dataset: DataSet, *args) -> Aggregator:
    return NoBlindingAggregator(sio, dataset, *args)

class NoBlindingAggregator(Aggregator):
    """Aggregates commits but not blinded and no range proofs"""

    def __init__(self, sio, dataset: DataSet):
        logger.info('Initializing client-side aggregator with secure aggregation without blinding values')
        super().__init__(sio, dataset)
        self.ci = CryptoInterface()
        self.register_at_server()

    def generate_training_finished_msg(self, update, msg):
        logger.info('Generate Pedersen commitments of update (without blindings)')
        update_enc = self.ci.commit_no_blinding(update)
        training_finished_msg = TrainingFinishedMsg(
            self.model.model_config.model_id,
            self.model.get_current_round_id(),
            content={
                'update_enc': update_enc
            }
        )
        return training_finished_msg

