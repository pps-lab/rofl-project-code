from os.path import join, exists, dirname, abspath
from threading import Thread
from subprocess import Popen

import sys
import time

SCRIPTS = dirname(dirname(abspath(__file__)))
ROOT = abspath(dirname(SCRIPTS))
sys.path.insert(0, ROOT)

import subprocess
import argparse

from test.config_loader import ConfigLoader

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

CONFIG = join(ROOT, 'config')
TEST = join(ROOT, 'test')
SRC = join(ROOT, 'fed_learning')
CRYPTO = join(SRC, 'crypto')
RUST_CRYPTO = join(CRYPTO, 'rust_crypto')

OLD_CONFIG_FILE = join(CONFIG, '.old.ini')
DATASPLITTER = join(TEST, 'data_splitter.py')
HELPER_SCRIPT = join(SCRIPTS, 'start_test_helper.sh')
TEST_SERVER = join(TEST, 'test_server.py')
TEST_CLIENT = join(TEST, 'test_client.py')

class LocalSimulationHelper(object):

    def __init__(self, config_file_path):
        self.config_file_path = abspath(config_file_path)
        self.config = ConfigLoader(config_file_path)
        logger.info('Running local simulation with %d clients' % self.config.num_clients)

        p = self._run_server()
        # thread = Thread(target=)
        # thread.start()
        self._run_clients_in_tab()

        p.wait()

    def _run_clients_in_tab(self):
        subcmds = [["python3", TEST_CLIENT, f"{x}", self.config_file_path] for x in range(1, self.config.num_clients + 1)]
        time.sleep(5)
        fs = [open(f"output_client_{i}.txt", "w") for i in range(1, self.config.num_clients + 1)]
        for c, f in zip(subcmds, fs):
            logger.info(f"{c} {f}")
            Popen(c, stdout=f)

    def _run_server(self):
        cmd = ['python3', TEST_SERVER, self.config_file_path]
        return Popen(cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Helper script for local simulation. Not meant to be run on its own!")
    parser.add_argument('config_file_path', type=str, help="Configuration file location")
    args = parser.parse_args()

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s in %(module)s: %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    sim = LocalSimulationHelper(args.config_file_path)