from os.path import join, exists, dirname, abspath
import sys

import subprocess
import argparse

from scripts.helper.util import *
from scripts.helper.util.proxy import Proxy
from scripts.helper.util.proxy_config_loader import ProxyConfigLoader

sys.path.insert(0, ROOT)
from test.config_loader import ConfigLoader

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class RemoteSimulation(object):

    def __init__(self, proxy_config_file_path, config_file_path, compile=False, distribute_data=False, pull=False):
        self.config_file_path = abspath(config_file_path)
        self.config = ConfigLoader(config_file_path)

        self.proxy_config_file_path = abspath(proxy_config_file_path)
        self.proxy_config = ProxyConfigLoader(self.proxy_config_file_path)
        self.proxy = Proxy(self.proxy_config, self.config)
        
        self.proxy.get_env()
        self.proxy.setup_project()
        if pull:
            self.proxy.pull_repo()
        self.proxy.transfer_config()
        self.proxy.setup_local_log_dir()
        self.proxy.bootstrap(compile=compile, distribute_data=distribute_data)
        self._run_remote_simulation()

    def _run_remote_simulation(self):
        sub_cmd = "python3 %s %s %s; bash" % (REMOTE_HELPER_SCRIPT, self.proxy_config_file_path, self.config_file_path)
        cmd = ['gnome-terminal', '--window', '--maximize', '--', 'bash', '-c', sub_cmd]
        subprocess.call(cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run remote simulation")
    parser.add_argument('deployment_config_file_path', type=str, help="Configuration file for deployment")
    parser.add_argument('config_file_path', type=str, help="Configuration file location")
    parser.add_argument('-r', '--reset', action='store_true', help="Redistribute data and recompile the rust crypto library")
    parser.add_argument('-d', '--data', action='store_true', help="Redistribute data")
    parser.add_argument('-c', '--compile', action='store_true', help="Recompile rust crypto library")
    parser.add_argument('-p', '--pull', action='store_true', help="Pull branch")
    args = parser.parse_args()

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s in %(module)s: %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)


    compile = args.compile or args.reset
    distribute_data = args.data or args.reset
    sim = RemoteSimulation(args.deployment_config_file_path, args.config_file_path, compile=compile, distribute_data=distribute_data, pull=args.pull)