from importlib import import_module

import sys
import pickle
import codecs
import socketio
import numpy as np
from keras.models import model_from_json

from fed_learning.client.local_model import LocalModel
from fed_learning.client.local_model_config import LocalModelConfig
from fed_learning.client.dataset import DataSet
from fed_learning.client.client_crypto_config import ClientCryptoConfig
from fed_learning.message import *

import logging

from test.demo.dev_fed_mnist_cnn_subspace import DevFedMNISTCNNSubspace

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Aggregator(object):

    def __init__(self, sio: socketio.Client, dataset: DataSet):
        self.dataset = dataset
        self.sio = sio
        self.register_handles()

    def register_handles(self):
        
        def on_connect(*args):
            pass

        def on_reconnect_attempt(*args):
            logger.info('Reconnect attempt')

        def on_reconnect_error(*args):
            logger.info('Reconnect error')

        def on_reconnect_failed(*args):
            logger.info('Reconnect failed')

        def on_transfer_model_config(*args):
            msg = parse_msg(args[0])
            self.setup_local_model_config(msg)
            logger.info('Received model configuration')
            self.sio.emit('received_model_config')
            
        def on_transfer_model(*args):
            msg = parse_msg(args[0])
            self.setup_local_model(msg)
            logger.info('Received model')
            self.sio.emit('received_model')

        def on_transfer_initial_weights(*args):
            msg = parse_msg(args[0])
            self.setup_initial_weights(msg)
            logger.info('Received initial weights')
            self.sio.emit('received_initial_weights')

        def on_transfer_crypto_config(*args):
            msg = parse_msg(args[0])
            logger.info('Received Crypto configuration')
            self.setup_crypto_config(msg)
            self.sio.emit('received_crypto_config')

        def on_disconnect(*args):
            logger.info('Terminating session')
            self.sio.disconnect()

        def on_transfer_weights(*args):
            msg = parse_msg(args[0])
            logger.info('received_weights')
            self.model.setup_new_round(msg.round_id, msg.weights)
            model_update = self.model.train_one_round()
            response_msg = self.generate_training_finished_msg(model_update, msg)
            logger.info('Update sent back')

            self.sio.emit('training_finished', response_msg.serialize())

        self.sio.on('connect', on_connect)
        self.sio.on('reconnect_attempt', on_reconnect_attempt)
        self.sio.on('reconnect_error', on_reconnect_error)
        self.sio.on('reconnect_failed', on_reconnect_failed)
        self.sio.on('transfer_model_config', on_transfer_model_config)
        self.sio.on('transfer_model', on_transfer_model)
        self.sio.on('transfer_initial_weights', on_transfer_initial_weights)
        self.sio.on('transfer_crypto_config', on_transfer_crypto_config)
        self.sio.on('transfer_weights', on_transfer_weights)
        self.sio.on('terminate_session', on_disconnect)

    def register_at_server(self):
        self.sio.emit('register_client')

    def setup_local_model_config(self, msg):
        model_config = LocalModelConfig(msg.num_of_clients,
                                        msg.client_batch_size,
                                        msg.num_local_epochs,
                                        msg.optimizer,
                                        msg.learning_rate,
                                        msg.loss,
                                        msg.metrics,
                                        msg.image_augmentation,
                                        msg.lr_decay,
                                        msg.model_id,
                                        msg.probabilistic_quantization,
                                        msg.fp_bits,
                                        msg.fp_frac,
                                        msg.range_bits)
        self.model = LocalModel(model_config, self.dataset)

    def setup_local_model(self, msg):
        if msg.model_name is not None:
            # Load model by name
            model_module = import_module(msg.model_name)
            model = model_module.getType().build_model(seed=msg.seed)
            self.model.set_model(model)
        else:
            self.model.set_model(model_from_json(msg.model))

    def setup_initial_weights(self, msg):
        """
        Setup
        :param msg: TransferInitialWeightsMsg
        :return:
        """
        self.model.set_all_weights(msg.weights)

    def setup_crypto_config(self, msg):
        crypto_config = ClientCryptoConfig(msg.value_range, msg.n_partition, msg.l2_value_range)
        self.crypto_config = crypto_config

    def generate_training_finished_msg(self, update, msg) -> TrainingFinishedMsg:
        raise NotImplementedError()
