from util import *

from os.path import join, exists, dirname, abspath
import sys

sys.path.insert(0, ROOT)

import argparse

from test.config_loader import ConfigLoader
from util.proxy import Proxy
from util.proxy_config_loader import ProxyConfigLoader

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Helper script for remote simulation. Not meant to be run on its own!")
    parser.add_argument('deployment_config_file_path', type=str, help="Configuration file for deployment")
    parser.add_argument('config_file_path', type=str, help="Configuration file location")
    args = parser.parse_args()

    config = ConfigLoader(args.config_file_path)

    proxy_config_file_path = abspath(args.deployment_config_file_path)
    proxy_config = ProxyConfigLoader(proxy_config_file_path)
    proxy = Proxy(proxy_config, config)
    proxy.run_server()