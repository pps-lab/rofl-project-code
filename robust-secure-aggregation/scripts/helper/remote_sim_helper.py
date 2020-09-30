from util import *

import os
from os.path import join, exists, dirname, abspath
import sys

sys.path.insert(0, ROOT)

import subprocess
import argparse

from test.config_loader import ConfigLoader
from util.proxy import Proxy
from util.proxy_config_loader import ProxyConfigLoader

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class RemoteSimulationHelper(object):

    def __init__(self, proxy_config_file_path, config_file_path):
        self.config_file_path = config_file_path
        self.config = ConfigLoader(config_file_path)

        self.proxy_config_file_path = abspath(proxy_config_file_path)
        self._run_clients_in_tab()
        self._run_server()


    def _run_clients_in_tab(self):
        raw_cmd = 'sleep 2; python3 %s %s %s %s; bash' 
        sub_cmd = [raw_cmd % (REMOTE_CLIENT_SCRIPT, x, self.proxy_config_file_path, self.config_file_path) for x in range(1, self.config.num_clients+1)]
        cmd = ['gnome-terminal', '--tab', '--', 'bash', '-c']
        for c in sub_cmd:
            subprocess.call(cmd + [c], cwd=SCRIPTS)

    def _run_server(self):
        cmd = ['python3', REMOTE_SERVER_SCRIPT, self.proxy_config_file_path, self.config_file_path]
        subprocess.call(cmd, cwd=SCRIPTS)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Helper script for remote simulation. Not meant to be run on its own!")
    parser.add_argument('deployment_config_file_path', type=str, help="Configuration file for deployment")
    parser.add_argument('config_file_path', type=str, help="Configuration file location")
    args = parser.parse_args()

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s in %(module)s: %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    sim = RemoteSimulationHelper(args.deployment_config_file_path, args.config_file_path)