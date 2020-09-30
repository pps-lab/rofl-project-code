import threading

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

import multiprocessing
from multiprocessing import Process

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

    def aggregate_updates(self) -> np.ndarray:
        client_updates_enc = [ x.content['update_enc'] for x in self.finished_clients_round.values() ]
        logger.info('Homomorphically adding weights')
        new_update_enc = self.ci.add_commitments(client_updates_enc)
        logger.info('Calculating Discrete Log')
        new_update_flat = self.ci.extract_values(new_update_enc)

        return new_update_flat

    def perform_checks(self):
        self.verify_randomness()


    def verify_randomness(self):
        rand_commits_list = []
        for sid, msg in self.finished_clients.items():
            randproofs = msg.content['randproofs']
            update_enc = msg.content['update_enc']
            rand_commits = msg.content['rand_commits']
            try:
                logger.info('Verifying randomness for client %s' % sid)
                # p = Process(target=self.ci.verify_randproof, args=(update_enc, rand_commits, randproofs))
                # p.start()
                # while p.is_alive():
                #     print(f"P {p}")
                #     self.socketio.sleep()
                # passed = True # ??
                # logger.info("Done verifying !")
                passed = self.ci.verify_randproof(update_enc, rand_commits, randproofs)
                if not passed:
                    logger.error('Randomness proof verification failed for client %s' % sid)
                    sys.exit(0)
                logger.info('Client %s passed randomness verification' % sid)
            except VerificationException as e:
                logger.error('Randomness proof verification threw exception for client %s with msg: %s' % (sid, str(e)))
                sys.exit(0)
            rand_commits_list.append(rand_commits)
        rand_sum = self.ci.add_commitments(rand_commits_list)
        if not self.ci.equals_neutral_group_element_vector(rand_sum):
            logger.error('Randomness does not cancel out')
            sys.exit(0)
