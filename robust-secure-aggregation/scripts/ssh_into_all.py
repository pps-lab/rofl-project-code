from os.path import join, exists, dirname, abspath
import sys

import time

import subprocess
import argparse

from helper.util.dist_proxy_config_loader import DistProxyConfigLoader

_DEFAULT_DP_CONF = '/home/mlei/Dropbox/mathias/studies/masterthesis/fed_learning/scripts/dist_config.ini'


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run remote simulation")
    parser.add_argument('--conf', type=str, default=_DEFAULT_DP_CONF, help="Configuration file for deployment")
    args = parser.parse_args()
    dpc_file_path = abspath(args.conf)

    dp_config = DistProxyConfigLoader(dpc_file_path)
    print(dp_config.clients)
    ssh_cmd_raw = 'ssh -i %s %s@%s'
    tab_cmd_raw = [
        'gnome-terminal',
        '--tab',
        '--',
        'bash', 
        '-c']
    
    for c in dp_config.clients:
        ssh_cmd = ssh_cmd_raw % (dp_config.key_file, dp_config.user, c)
        tab_cmd = tab_cmd_raw + [ssh_cmd]
        print(subprocess.run(tab_cmd))


    ssh_cmd_server = ssh_cmd_raw % (dp_config.key_file, dp_config.user, dp_config.server)
    tab_cmd = tab_cmd_raw + [ssh_cmd_server]
    print(subprocess.run(tab_cmd))

