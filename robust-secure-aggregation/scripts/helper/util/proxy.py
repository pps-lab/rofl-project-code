import os
from os.path import abspath, dirname, join, exists
from ntpath import basename
import time
import subprocess

from . import *
from .proxy_config_loader import ProxyConfigLoader

from fabric.connection import Connection

class Proxy(object):
    
    def __init__(self, proxy_config: ProxyConfigLoader, local_config: ConfigLoader, compile=False, distribute_data=False):
        self.pc = proxy_config
        self.lc = local_config
        self.target_folder = self.pc.target_folder
        self.root = join(self.target_folder, 'fed_learning')
        self.config_dir = join(self.root, 'config')
        self.test_dir = join(self.root, 'test')
        self.log_dir = join(self.root, 'log')
        self.scripts_dir = join(self.root, 'scripts')
        self.helper_dir = join(self.scripts_dir, 'helper')
        self.remote_config_path = join(self.config_dir, basename(self.lc.config_file_path))

        self.setup_script = join(self.scripts_dir, 'setup_venv.sh')
        self.bootstrap_script = join(self.helper_dir, 'bootstrap.py')
        self.test_server = join(self.test_dir, 'test_server.py')
        self.test_client = join(self.test_dir, 'test_client.py')
        self.data_splitter = join(self.test_dir, 'data_splitter.py')

        self.local_log_dir = join(LOG, 'remote')

        connect_kwargs = {"key_filename":self.pc.key_file}
        self.c = Connection(self.pc.host,
                            user=self.pc.user,
                            connect_kwargs=connect_kwargs
                            )

    def setup_project(self):
        if not self.project_exists():
            with self.c.cd(self.target_folder):
                self.c.run('git clone -b %s %s' % (self.pc.branch, self.pc.repo))
                self.c.sudo('chmod 755 %s' % self.setup_script)
                self.c.run(self.setup_script)

    def pull_repo(self):
        if not self.project_exists():
            print('Error: project does not exist yet, clone the repo first!')
            return
        with self.c.cd(self.root):
            self.c.run('git pull origin %s' % self.pc.branch.rstrip('\n'))

    def bootstrap(self, compile=False, distribute_data=False):
        if not self.project_exists():
            print('Error: project does not exist yet, clone the repo first!')
            return
        cmd = 'python3 %s %s' % (self.bootstrap_script, self.remote_config_path)
        cargo_path_var = "PATH=$PATH:~/.cargo/bin ; "
        if compile: 
            cmd += ' -c'
        if distribute_data:
            cmd += ' -d'
        self.c.run(cargo_path_var + cmd)
        
    def transfer_config(self):
        remote_config_dir = self.c.run('readlink -e %s' % self.config_dir).stdout.rstrip('\n')
        self.c.put(self.lc.config_file_path, remote_config_dir)

    def run_server(self):
        if not self.project_exists():
            print('Error: project does not exist yet, clone the repo first!')
            return
        self.c.run('python3 %s %s' % (self.test_server, self.remote_config_path))
        file_name = 'test_server.log'
        #NOTE mlei: underlying sftp server cannot handle ~ in filepaths
        remote_log_file = self.c.run('readlink -e %s' % join(self.log_dir, file_name)).stdout.rstrip('\n\r')
        local_log_file = join(self.local_log_dir, file_name)
        print('retrieving %s to %s' % (remote_log_file, local_log_file))
        self.c.get(remote_log_file, local_log_file)
    
    def run_client(self, id):
        if not self.project_exists():
            print('Error: project does not exist yet, clone the repo first!')
            return
        self.c.run('python3 %s %s %s' % (self.test_client, id, self.remote_config_path))
        file_name = 'test_client_%s.log' % id
        remote_log_file = self.c.run('readlink -e %s' % join(self.log_dir, file_name)).stdout.rstrip('\n\r')
        local_log_file = join(self.local_log_dir, file_name)
        print('retrieving %s to %s' % (remote_log_file, local_log_file))
        self.c.get(str(remote_log_file), local_log_file)

    def run_data_splitter(self):
        if not self.project_exists():
            print('Error: project does not exist yet, clone the repo first!')
            return
        self.c.run('python3 %s %s' % (self.data_splitter, self.remote_config_path))

    def get_env(self):
        self.c.run('env')

    def project_exists(self):
        return self.exists(self.root)

    def exists(self, path):
        cmd = 'test -e %s' % path
        return self.c.run(cmd, hide=True, warn=True).ok

    def setup_local_log_dir(self):
        if os.path.exists(self.local_log_dir):
            cmd = ['rm', '-rf', self.local_log_dir]
            subprocess.call(cmd)
        os.mkdir(self.local_log_dir)            

