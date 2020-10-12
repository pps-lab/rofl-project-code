import sys
import pickle
import codecs
import uuid
from threading import Lock
import numpy as np

from flask import request
from flask_socketio import SocketIO, disconnect
from flask_socketio import *

from fed_learning.server.aggregator import Aggregator
from fed_learning.server.aggregator.secure_aggregator import SecureAggregator
from fed_learning.server.global_model import GlobalModel
from fed_learning.message import *
from fed_learning.crypto.crypto_interface import CryptoInterface
from fed_learning.crypto.crypto_interface.exception import VerificationException

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def build_aggregator(sio: SocketIO, global_model: GlobalModel, *args) -> Aggregator:
    return PartialRangeAggregator(sio, global_model, *args)

class PartialRangeAggregator(SecureAggregator):
    
    def __init__(self, socketio: SocketIO, global_model: GlobalModel, *args):
        super().__init__(socketio, global_model, *args)
        logger.info('Augmenting secure aggregation with rangeproofs')
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

    def aggregate_updates(self) -> np.ndarray:
        client_updates_enc = [ x.content['update_enc'] for x in self.finished_clients_round.values() ]
        logger.info('Homomorphically adding weights')
        new_update_enc = self.ci.add_commitments(client_updates_enc)
        logger.info('Calculating Discrete Log')
        new_update_flat = self.ci.extract_values(new_update_enc)
        return new_update_flat

    def perform_checks(self):
        self.verify_randomness()
        self.verify_range()

    def verify_range(self):
        random_seed = 420

        for sid, msg in self.finished_clients.items():
            update_enc = msg.content['update_enc']
            rangeproofs = msg.content['rangeproofs']

            total_number_of_updates = self.global_model.params_count()
            np.random.seed(random_seed)
            selected_indices = np.random.choice(range(0, total_number_of_updates), int(total_number_of_updates * 0.005), replace=False)
            commits_to_verify = self.ci.select_commitments(update_enc, selected_indices)

            try:
                logger.info('Verifying range for client %s (%d range)' % (sid, self.crypto_config.value_range))
                passed = self.ci.verify_rangeproof(commits_to_verify, rangeproofs, self.crypto_config.value_range)
                if not passed:
                    logger.warn('Range proof verification failed for client %s' % sid)
                    sys.exit(0)
                logger.info('Client %s passed range verification' % sid)
            except VerificationException as e:
                    logger.warn('Range proof verification threw exception for client %s with msg: %s' % (sid, str(e)))
                    sys.exit(0)
