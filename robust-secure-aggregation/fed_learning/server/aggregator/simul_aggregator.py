import sys
import pickle
import codecs
import uuid
from threading import Lock
import numpy as np

from flask import request
from flask_socketio import SocketIO, disconnect


from fed_learning.server.aggregator import Aggregator
from fed_learning.server.global_model import GlobalModel
from fed_learning.message import *
from fed_learning.crypto.crypto_interface import CryptoInterface
from fed_learning.crypto.crypto_interface.exception import VerificationException


import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def build_aggregator(sio: SocketIO, global_model: GlobalModel, *args) -> Aggregator:
    return SecureAggregator(sio, global_model, *args)


class SecureAggregator(Aggregator):

    def __init__(self, socketio: SocketIO, global_model: GlobalModel, *args):
        logger.info('Initializing server-side aggregator with secure aggregation')
        super().__init__(socketio, global_model, *args)
        self.ci = CryptoInterface()

    def generate_training_params(self) -> dict:
        cancelling_blindings = self.ci.generate_cancelling_blindings(
            self.global_model.model_config.num_clients,
            self.global_model.params_count())
        msg_dict = {}
        for sid, blindings in zip(self.registered_clients.keys(), cancelling_blindings):

            msg_dict[sid] = StartTrainingMsg(
                self.global_model.get_current_round_id(),
                self.global_model.get_weights(),
                content={'blindings': blindings}
            )
            self.pending_clients.add(sid)
        return msg_dict

    def perform_checks(self):
        pass

    def aggregate_updates(self) -> np.ndarray:
        # self.verify_randomness()
        client_updates_enc = [ x.content['update_enc'] for x in self.finished_clients_round.values() ]
        logger.info('Homomorphically adding weights')
        new_update_enc = self.ci.add_commitments(client_updates_enc)
        logger.info('Calculating Discrete Log')
        new_update_flat = self.ci.extract_values(new_update_enc)
        return new_update_flat