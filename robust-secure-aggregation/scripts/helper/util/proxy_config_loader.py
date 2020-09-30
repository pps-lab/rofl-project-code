import os
from os.path import dirname, abspath, join, exists
import configparser
import pprint

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class ProxyConfigLoader(object):

    def __init__(self, config_file_path, log_summary=False ):
        self.config_file_path = config_file_path
        
        self.parser = configparser.ConfigParser()
        if not exists(self.config_file_path):
            logger.error('Proxy Configuration file not found: %s' % self.config_file_path )
            exit(-1)
        self.parser.read(self.config_file_path)

        logger.info('Loading Configuration file: %s' % self.config_file_path)

        self.host = self.parser.get('connection', 'HOST')
        self.user = self.parser.get('connection', 'USER')
        self.key_file = self.parser.get('connection', 'KEY_FILE')

        self.repo = self.parser.get('repository', 'REPO')
        self.branch = self.parser.get('repository', 'BRANCH')
        self.target_folder = self.parser.get('repository', 'TARGET_FOLDER')
        self.repo_user = self.parser.get('repository', 'REPO_USER')

        #self.cargo = self.parser.get('util', 'CARGO')

        if log_summary:
            logger.info('Loaded Configuration file:')
            formatted = pprint.pformat(self.__dict__)
            for line in formatted.splitlines():
                logger.info(line.rstrip())

    def _get_list(self, option, sep=',', chars=None):
        return [ chunk.strip(chars) for chunk in option.split(sep) ]