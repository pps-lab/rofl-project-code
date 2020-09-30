

# import eventlet
# eventlet.monkey_patch()
# from gevent import monkey
# monkey.patch_all()

import time

from fed_learning.util import async_tools
import sys
import logging
import json
import pickle
import codecs
import tensorflow as tf

import socketio
from keras.models import model_from_json

from fed_learning.client.local_model import LocalModel
from fed_learning.client.local_model_config import LocalModelConfig
from fed_learning.client.dataset import DataSet
from fed_learning.client.aggregator.plain_aggregator import PlainAggregator, build_aggregator
from fed_learning.client.aggregator.no_blinding_aggregator import NoBlindingAggregator

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Client(object):
    
    def __init__(self, server_host, server_port, id, dataset: DataSet, aggregator_builder=build_aggregator):
        self.id = id
        self.dataset = dataset
        print("Initializing")
        logger.info('Initializing client with id ' + str(self.id))

        time.sleep(5)

        self.sio = socketio.Client(reconnection_delay_max=10000, request_timeout=3600)
        self.url = 'http://%s:%d' % (server_host, server_port)

        print(f"Will connect to {self.url}", flush=True)

        self.sio.connect(self.url)
        # tries = 10
        # while tries > 0:
        #     try:
        #         self.sio.connect(self.url)
        #     except:
        #         logger.info(f"Failed to connect to server. Will retry in 1 second ({tries})")
        #         tries -= 1
        #         time.sleep(1)
        # if tries <= 0:
        #     logger.error("Failed to connect to server. Exiting.")
        #     sys.exit(1)
        logger.info('Client connected to %s:%d', server_host, server_port)

        self.aggregator = aggregator_builder(self.sio, dataset)
        self.sio.wait()
        logger.info('Exiting')
        sys.exit(0)

