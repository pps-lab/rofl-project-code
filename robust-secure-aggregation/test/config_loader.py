import os
from os.path import dirname, abspath, join, exists
import configparser
import pprint

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_ROOT = dirname(dirname(abspath(__file__)))
_DEFAULT_CONFIG = join(_ROOT, 'config', 'dev_config.ini')
# _DEFAULT_CONFIG = join(_ROOT, 'config', 'dev_log_reg_config.ini')

class ConfigLoader(object):

    def __init__(self, config_file_path=_DEFAULT_CONFIG, log_summary=True ):
        self.config_file_path = config_file_path
        
        self.parser = configparser.ConfigParser()
        if not exists(self.config_file_path):
            logger.error('Configuration file not found: %s' % self.config_file_path )
            exit(-1)
        self.parser.read(self.config_file_path)

        print("Loading ", self.config_file_path)
        logger.info('Loading Configuration file: %s' % self.config_file_path)

        self.host = self.parser.get('connection', 'HOST')
        self.port = self.parser.getint('connection', 'PORT')

        self.dataset = self.parser.get('dataset', 'DATASET')
        self.transformer = self.parser.get('dataset', 'TRANSFORMER', fallback=None)
        self.training_size = self.parser.getint('dataset', 'TRAINING_SIZE', fallback=None)
        self.test_size = self.parser.getint('dataset', 'TEST_SIZE', fallback=None)
        self.split_mode = self.parser.get('dataset', 'SPLIT_MODE', fallback='even')
       
        self.model = self.parser.get('model', 'MODEL')
        self.iterations = self.parser.getint('model', 'ITERATIONS')
        self.num_clients = self.parser.getint('model', 'NUM_CLIENTS')
        self.client_batch_size = self.parser.getint('model', 'CLIENT_BATCH_SIZE')
        self.num_local_epochs = self.parser.getint('model', 'NUM_LOCAL_EPOCHS')
        self.optimizer = self.parser.get('model', 'OPTIMIZER')
        self.learning_rate = self.parser.getfloat('model', 'LEARNING_RATE')
        self.loss = self.parser.get('model', 'LOSS')
        self.metrics = self._get_list(self.parser.get('model', 'METRICS'))
        self.image_augmentation = self.parser.get('model', 'IMAGE_AUGMENTATION', fallback=False)
        self.lr_decay = self.parser.get('model', 'LEARNING_RATE_DECAY', fallback=None)

        self.aggregator = self.parser.get('aggregator', 'AGGREGATOR')

        self.fp_bits = self.parser.getint('crypto', 'FP_BITS')
        self.fp_frac = self.parser.getint('crypto', 'FP_FRAC')
        self.n_partition = self.parser.getint('crypto', 'N_PARTITION')
        self.value_range = self.parser.getint('crypto', 'VALUE_RANGE')
        self.l2_value_range = self.parser.getint('crypto', 'L2_VALUE_RANGE')
        self.probabilistic_quantization = self.parser.getboolean('crypto', 'PROBABILISTIC_QUANTIZATION')
        self.optimistic_starting = self.parser.getboolean('crypto', 'OPTIMISTIC_STARTING', fallback=False)

        if self.fp_bits < self.fp_frac:
            logger.error('Bitsize for fixed point precision cannot be smaller than the number of fractional bits demanded')
            exit(-1)

        log = self.parser.get('log', 'LOG', fallback=None)
        self.log_path = join(_ROOT, log) if log is not None else None

        if log_summary:
            logger.info('Loaded Configuration file:')
            formatted = pprint.pformat(self.__dict__)
            for line in formatted.splitlines():
                logger.info(line.rstrip())

    def _get_list(self, option, sep=',', chars=None):
        return [ chunk.strip(chars) for chunk in option.split(sep) ]