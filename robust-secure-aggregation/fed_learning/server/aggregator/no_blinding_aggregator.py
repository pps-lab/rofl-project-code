import pickle
import codecs
import uuid
from threading import Lock
import numpy as np

from flask import request
from flask_socketio import SocketIO, disconnect
from flask_socketio import *

from fed_learning.server.aggregator import Aggregator
from fed_learning.server.global_model import GlobalModel
from fed_learning.message import *
from fed_learning.crypto.crypto_interface import CryptoInterface

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def build_aggregator(sio: SocketIO, global_model: GlobalModel, *args) -> Aggregator:
    return NoBlindingAggregator(sio, global_model, *args)

class NoBlindingAggregator(Aggregator):
    
    def __init__(self, socketio: SocketIO, global_model: GlobalModel, *args):
        logger.info('Initializing server-side aggregator with secure aggregation without blinding values')
        super().__init__(socketio, global_model, *args)
        self.ci = CryptoInterface()

    def generate_training_params(self) -> dict:
        msg_dict = {}
        for sid in self.registered_clients:

            msg_dict[sid] = StartTrainingMsg(
                self.global_model.get_current_round_id(),
                self.global_model.get_weights()
            )
            self.pending_clients.add(sid)
        return msg_dict

    def aggregate_updates(self) -> np.ndarray:
        client_updates_enc = [ x.content['update_enc'] for x in self.finished_clients_round.values() ]
        logger.info('Homomorphically adding weights')
        new_update_enc = self.ci.add_commitments(client_updates_enc)
        logger.info('Calculating Discrete Log')
        new_update_flat = self.ci.extract_values(new_update_enc)
        return new_update_flat

    def perform_checks(self):
        pass
