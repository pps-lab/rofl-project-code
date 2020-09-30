import os
import threading
import time

import sys
import pickle
import codecs
import uuid
from threading import Lock
#import threading
# from eventlet import tpool
import numpy as np

from flask import request
from flask_socketio import SocketIO, disconnect
from flask_socketio import *

from fed_learning.util.async_tools import ASYNC_MODE, run_native, shutdown_pool
from fed_learning.server.global_model import GlobalModel
from fed_learning.message import *
from fed_learning.server.server_crypto_config import ServerCryptoConfig

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Aggregator(object):

    def __init__(self, socketio: SocketIO, global_model: GlobalModel, crypto_config: ServerCryptoConfig):
        self.socketio = socketio
        self.global_model = global_model
        self.crypto_config = crypto_config
        self.registered_clients = {}
        self.pending_clients = set()
        self.finished_clients = {}
        self.finished_clients_round = {}
        self.is_training = False
        self.global_model.evaluate()
        self.register_handles()
        self.checking_previous_round = False
        self.optimistic_starting = True

    def register_handles(self):

        @self.socketio.on('connect')
        def handle_connect():
            pass

        @self.socketio.on('register_client')
        def handle_register_client():
            logger.info(f"Client {request.sid} registered.")
            if request.sid not in self.registered_clients:
                self._transfer_model_config(request.sid)
            else:
                logger.warn(str(request.sid) + ' attempted to connect twice')

        @self.socketio.on('received_model_config')
        def handle_received_model_config():
            self._transfer_model(request.sid)

        @self.socketio.on('received_model')
        def handle_received_model():
            initial_weights = self.global_model.get_all_weights()
            if initial_weights is not None:
                self._transfer_initial_weights(initial_weights, request.sid)
            else:
                self._transfer_crypto_config(request.sid)

        @self.socketio.on('received_initial_weights')
        def handle_received_model():
            self._transfer_crypto_config(request.sid)

        @self.socketio.on('received_crypto_config')
        def handle_received_crypto_config():
            assert(request.sid not in self.registered_clients)
            self.registered_clients[request.sid] = ''

            # enough  participants -> start training
            if len(self.registered_clients) >= self.global_model.model_config.num_clients and not self.is_training:
                self.start_training_round()

        @self.socketio.on('disconnect')
        def handle_disconnect():
            try:
                del self.registered_clients[request.sid]
                logger.info(f"{request.sid} has disconnected")
                # TODO mlei: handle disconnect during training
            except KeyError:
                logger.error('Disconnect from non-registered user')

        @self.socketio.on_error_default  # handles all namespaces without an explicit error handler
        def default_error_handler(e):
            logger.error("Error occurred")
            logger.error(e)

        @self.socketio.on('training_finished')
        def handle_training_finished(content):
            if request.sid not in self.pending_clients:
                logger.warn('Received update from invalid client')
                return
            # logger.info(f"fun start: {time.time()}")
            # logger.info("Got training finished on thread %s" % threading.current_thread())
            msg = parse_msg(content)

            if not msg.model_id == self.global_model.model_config.model_id:
                logger.warn('Received update with wrong model id')
                return

            if not msg.round_id == self.global_model.get_current_round_id():
                logger.warn('Received update with wrong round id')
                return

            try:
                self.pending_clients.remove(request.sid)
            except KeyError:
                logger.error('Client id missing in pending clients')
                return
            # store the message with client id
            self.finished_clients_round[request.sid] = msg
            if not self.pending_clients:
                # all responses received
                assert(len(self.finished_clients_round) == self.global_model.model_config.num_clients)
                self.finish_training_round()

    def _transfer_model_config(self, sid):
        config = self.global_model.model_config
        msg = TransferModelConfigMsg(config.num_clients,
                                     config.client_batch_size,
                                     config.num_local_epochs,
                                     config.optimizer,
                                     config.learning_rate,
                                     config.loss,
                                     config.metrics,
                                     config.model_id,
                                     config.probabilistic_quantization,
                                     config.fp_bits,
                                     config.fp_frac,
                                     config.range_bits)
        self.socketio.emit('transfer_model_config', msg.serialize(), room=sid)

    def _transfer_model(self, sid):
        if self.global_model.serialize_model_name() is not None:
            seed = self.global_model.get_weights_seed()
            msg = TransferModelMsg(model_name=self.global_model.serialize_model_name(), seed=seed)
        else:
            msg = TransferModelMsg(model=self.global_model.serialize_model())
        self.socketio.emit('transfer_model', msg.serialize(), room=sid)

    def _transfer_initial_weights(self, weights, sid):
        msg = TransferInitialWeightsMsg(weights)
        new_weights = weights
        self.socketio.emit('transfer_initial_weights', msg.serialize(), room=sid)

    def _transfer_crypto_config(self, sid):
        msg = TransferCryptoConfigMsg(self.crypto_config.value_range,
                                      self.crypto_config.n_partition,
                                      self.crypto_config.l2_value_range)
        self.socketio.emit('transfer_crypto_config', msg.serialize(), room=sid)

    def _broadcast_termination(self):
        logger.info('Broadcasting termination')
        for client in self.registered_clients:
            self.socketio.emit('terminate_session', room=client)

    def terminate_session(self):
        logger.info('Terminating session')
        self._broadcast_termination()
        self.socketio.sleep(1)
        #logging.info('Calling exit from thread %s' % threading.current_thread())
        #pool.killall()
        #self.socketio.sleep(1)
        shutdown_pool()
        os._exit(0)

    def start_training_round(self):
        self.is_training = True
        self.global_model.start_training_round()
        logger.info('Starting training round nr %d', self.global_model.get_round_nr())
        self.pending_clients = set()
        if self.optimistic_starting:
            # Copy to do the checking
            self.finished_clients = self.finished_clients_round.copy()
        self.finished_clients_round = {}
        msg_dict = self.generate_training_params()
        self.broadcast_start_training(msg_dict)

    def generate_training_params(self) -> dict:
        raise NotImplementedError()

    def aggregate_updates(self) -> np.ndarray:
        raise NotImplementedError()

    def perform_checks(self):
        raise NotImplementedError()

    def perform_checks_around(self):
        """ Handles the server logic so the aggregators dont have to"""
        self.perform_checks()
        self.checking_previous_round = False
        self.finished_clients = {}
        logger.info("Done with checks")

    def broadcast_start_training(self, msg_dict):
        for client in self.registered_clients:
            try:
                msg = msg_dict[client]
            except KeyError as e:
                logger.error('Message for client %s  could not be found' % client)
                raise e
            self.pending_clients.add(client)
            self.socketio.emit('transfer_weights', msg.serialize(), room=client)

    def finish_training_round(self):
        logger.info('Aggregating weights')
        while self.checking_previous_round:
            logger.info("Still verifying the previous round.")
            self.socketio.sleep(1)

        if not self.optimistic_starting:
            self.finished_clients = self.finished_clients_round # just reference as we will still have it in memory
            self.perform_checks_around()

        aggregated_update = self.aggregate_updates()

        logger.info("Done aggregating")

        self.global_model.update_weights(aggregated_update/self.global_model.model_config.num_clients)
        self.global_model.evaluate()


        logger.info('Finished training round nr %d', self.global_model.get_round_nr())
        if self.global_model.get_round_nr() < self.global_model.model_config.iterations:
            self.start_training_round()
        else:
            logger.info('Training finished')
            self.terminate_session()

        if self.optimistic_starting:
            self.checking_previous_round = True
            self.socketio.start_background_task(target=self.perform_checks_around)
        # v = Verifier(self.verifier_queue, self.finished_clients)
        # v.start()
        # self.perform_checks_around()
        # t = threading.Thread(target=self.perform_checks_around, args=())
        # t.start()
