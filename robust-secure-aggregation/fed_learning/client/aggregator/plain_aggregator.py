import sys
import pickle
import codecs
from keras.models import model_from_json

from fed_learning.client.aggregator import Aggregator
from fed_learning.client.local_model import LocalModel
from fed_learning.client.local_model_config import LocalModelConfig
from fed_learning.client.dataset import DataSet
from fed_learning.message import *


import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def build_aggregator(sio, dataset: DataSet, *args) -> Aggregator:
    return PlainAggregator(sio, dataset, *args)

class PlainAggregator(Aggregator):
    """Aggregates without anything, not even clipping"""

    def __init__(self, sio, dataset: DataSet):
        logger.info('Initializing client-side aggregator without encryption')
        super().__init__(sio, dataset)
        self.register_at_server()

    def generate_training_finished_msg(self, update, msg):
        logger.info('Generating TrainingFinishedMsg without encryption')
        training_finished_msg = TrainingFinishedMsg(
            self.model.model_config.model_id,
            self.model.get_current_round_id(),
            content={
                'update': update
            }
        )
        return training_finished_msg
