from fed_learning.util.async_tools import ASYNC_MODE
import sys
import pickle
from threading import Lock

# import eventlet
# eventlet.monkey_patch()
# from gevent import monkey
# monkey.patch_all()

from flask import *
from flask_socketio import SocketIO
from flask_socketio import *
import logging

from rig.type_casts import float_to_fp, float_to_fix

from fed_learning.server.global_model import GlobalModel
from fed_learning.server.aggregator.plain_aggregator import PlainAggregator, build_aggregator
from fed_learning.server.aggregator.no_blinding_aggregator import NoBlindingAggregator
from fed_learning.server.server_crypto_config import ServerCryptoConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Server(object):

    def __init__(self,
                 host,
                 port,
                 global_model: GlobalModel,
                 crypto_config: ServerCryptoConfig,
                 aggregator_builder=build_aggregator):
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app, async_mode=ASYNC_MODE, logger=logger, engineio_logger=True, ping_interval=3600, ping_timeout=172800,
                                 monitor_clients=False)

        self.host = host
        self.port = port

        self.aggregator = aggregator_builder(self.socketio, global_model, crypto_config)

    def start(self):
        logger.info('Server listening at %s:%d', self.host, self.port)
        self.socketio.run(self.app, host=self.host, port=self.port)
        logger.info('Exiting')
        sys.exit(0)
