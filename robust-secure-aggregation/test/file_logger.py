import os
from os.path import join, abspath, dirname, isabs
import shutil
import logging
from logging import Logger
import tensorflow as tf

from test.config_loader import ConfigLoader

# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
ROOT = abspath(dirname(__file__))

def setup_file_logger(logger: Logger, config: ConfigLoader, file_name: str):

        logging.getLogger('socket').setLevel(logging.WARN)
        logging.getLogger('socketio').setLevel(logging.WARN)
        logging.getLogger('engineio').setLevel(logging.WARN)
        
        # NOTE mlei: nothing seems to change the tensorflow log level...
        tf.logging.set_verbosity(tf.logging.INFO)
        logging.getLogger("tensorflow").setLevel(logging.INFO)

        if isabs(config.log_path):
            log_path = config.log_path
        else:
            log_path = join(ROOT, config.log_path)

        log_file_path = join(log_path, file_name)
        if config.log_path is not None:
            if not os.path.exists(log_path):
                os.mkdir(log_path)
            else:
                if os.path.exists(log_file_path):
                    os.remove(log_file_path)

            logger.info('Log file path specified: %s', log_path)
            logger.setLevel(logging.INFO)
            fh = logging.FileHandler(log_file_path)
            #fh.setLevel(logging.INFO)
            formatter = logging.Formatter('[%(asctime)s] %(levelname)s in %(module)s: %(message)s')
            fh.setFormatter(formatter)
            logger.addHandler(fh)

        # try:
        #     import watchtower
        #     print("Setting up AWS CloudWatch logging")
        #     handler = watchtower.CloudWatchLogHandler(stream_name=file_name)
        #     logger.addHandler(handler)
        # except ImportError as e:
        #     pass  # module doesn't exist, deal with it.


