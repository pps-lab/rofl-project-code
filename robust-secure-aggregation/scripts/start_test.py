
from gevent import monkey
monkey.patch_all()

from os.path import join, exists, dirname, abspath
import sys

from helper.util import *


import subprocess
import argparse

sys.path.insert(0, ROOT)
from test.config_loader import ConfigLoader


import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class LocalSimulation(object):

    def __init__(self, config_file_path, compile=False, distribute_data=False):
        self.config_file_path = abspath(config_file_path)
        self.config = ConfigLoader(config_file_path)

        bootstrap(self.config, compile=compile, distribute_data=distribute_data)
        self._run_local_simulation()

    def _run_local_simulation(self):
        sub_cmd = "conda activate robust-secure-aggregation; python3 %s %s; bash" % (LOCAL_HELPER_SCRIPT, self.config_file_path)
        # cmd = ['sh', sub_cmd]
        # cmd = ['gnome-terminal', '--window', '--maximize', '--', 'bash', '-c', sub_cmd]
        cmd = ['ttab', '-w', '-a', 'iTerm2', sub_cmd]
        subprocess.call(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run local simulation")
    parser.add_argument('config_file_path', type=str, help="Configuration file location")
    parser.add_argument('-r', '--reset', action='store_true', help="Redistribute data and recompile the rust crypto library")
    parser.add_argument('-d', '--data', action='store_true', help="Redistribute data")
    parser.add_argument('-c', '--compile', action='store_true', help="Recompile rust crypto library")
    args = parser.parse_args()

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s in %(module)s: %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    compile = args.compile or args.reset
    distribute_data = args.data or args.reset
    sim = LocalSimulation(args.config_file_path, compile=compile, distribute_data=distribute_data)