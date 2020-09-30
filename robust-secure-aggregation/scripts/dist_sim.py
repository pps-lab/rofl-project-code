from os.path import join, exists, dirname, abspath
import sys

import subprocess
import argparse

from scripts.helper.util import *
from scripts.helper.dist_proxy import DistProxy
from scripts.helper.util.dist_proxy_config_loader import DistProxyConfigLoader

sys.path.insert(0, ROOT)
from test.config_loader import ConfigLoader

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_DEFAULT_DP_CONF = '/home/mlei/Dropbox/mathias/studies/masterthesis/fed_learning/scripts/dist_config.ini'
_DEFAULT_CONF = '/home/mlei/Dropbox/mathias/studies/masterthesis/fed_learning/config/dist_log_reg_config.ini'
class DistSim(object):

    def __init__(
        self, 
        dpc_file_path=_DEFAULT_DP_CONF, 
        config_file_path=_DEFAULT_CONF, 
        compile=False, 
        distribute_data=False, 
        pull=False):
        self.config_file_path = abspath(config_file_path)
        self.config = ConfigLoader(config_file_path)

        self.dpc_file_path = abspath(dpc_file_path)
        self.dp_config = DistProxyConfigLoader(self.dpc_file_path)
        self.proxy = DistProxy(self.dp_config, self.config)
        
        print(self.proxy.dpc._clients)
        self.proxy.say_hello_all_hosts()

        self.proxy.get_env()
        self.proxy.setup_project()
        if pull:
            self.proxy.pull_repo()
        self.proxy.transfer_config()
        self.proxy.setup_local_log_dir()
        self.proxy.bootstrap(compile=compile, distribute_data=distribute_data)
        self._run_remote_simulation()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run remote simulation")
    parser.add_argument('--dpath', type=str, default=_DEFAULT_DP_CONF, help="Configuration file for deployment")
    parser.add_argument('--cpath', type=str, default=_DEFAULT_CONF, help="Configuration file location")
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
    sim = DistSim(args.dpath, args.cpath, compile=compile, distribute_data=distribute_data, pull=args.pull)