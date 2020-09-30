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
from fed_learning.server.aggregator.range_aggregator import RangeAggregator
from fed_learning.server.global_model import GlobalModel
from fed_learning.message import *
from fed_learning.crypto.crypto_interface import CryptoInterface
from fed_learning.crypto.crypto_interface.exception import VerificationException

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def build_aggregator(sio: SocketIO, global_model: GlobalModel, *args) -> Aggregator:
    return L2Aggregator(sio, global_model, *args)

class L2Aggregator(RangeAggregator):
    
    def __init__(self, socketio: SocketIO, global_model: GlobalModel, *args):
        super().__init__(socketio, global_model, *args)
        logger.info('Augmenting secure aggregation with l2 range proof')
        self.ci = CryptoInterface()

    def verify_randomness(self):
        rand_commits_list = []
        for sid, msg in self.finished_clients.items():
            self.socketio.sleep()
            randproofs = msg.content['randproofs']
            update_enc = msg.content['update_enc']
            rand_commits = msg.content['rand_commits']
            square_commits = msg.content['square_commits']
            l2_rangeproof = msg.content['l2_rangeproof']
            square_sum_commit = msg.content['square_sum_commit']
            l2_range_exp = self.crypto_config.l2_value_range
            try:
                logger.info('Verifying l2 range proof and squared randomness for client %s with range %s' % (sid, l2_range_exp))
                passed = self.ci.verify_l2proof(update_enc, rand_commits, square_commits, randproofs, l2_rangeproof, square_sum_commit, l2_range_exp)
                if not passed:
                    logger.error('L2 range proof and squared randomness proof verification failed for client %s' % sid)
                    sys.exit(0)
                logger.info('Client %s passed randomness verification' % sid)
            except VerificationException as e:
                logger.error('L2 range proof and squared randomness proof verification'
                             ' threw exception for client %s with msg: %s' % (sid, str(e)))
                sys.exit(0)
            rand_commits_list.append(rand_commits)
        rand_sum = self.ci.add_commitments(rand_commits_list)
        if not self.ci.equals_neutral_group_element_vector(rand_sum):
            logger.error('Randomness does not cancel out')
            sys.exit(0)

