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

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def build_aggregator(sio: SocketIO, global_model: GlobalModel, *args) -> Aggregator:
    return PlainAggregator(sio, global_model, *args)

class PlainAggregator(Aggregator):

    def __init__(self, socketio: SocketIO, global_model: GlobalModel, *args):
        logger.info('Initializing server-side aggregator without any encryption')
        super().__init__(socketio, global_model, *args)

    def generate_training_params(self) -> dict:
        msg_dict = {}
        for sid in self.registered_clients:

            msg_dict[sid] = StartTrainingMsg(
                self.global_model.get_current_round_id(),
                self.global_model.get_weights()
            )
            self.pending_clients.add(sid)
        return msg_dict

    def aggregate_updates(self):
        client_updates = [ x.content['update'] for x in self.finished_clients_round.values() ]
        new_update = sum(client_updates, 0)
        return new_update

    def perform_checks(self):
        # None!
        pass
