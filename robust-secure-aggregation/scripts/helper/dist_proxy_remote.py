import os
from os.path import abspath, dirname, join, exists
import sys
from ntpath import basename
import time
import subprocess
import argparse
from collections import defaultdict

from scripts.helper.util import *
from scripts.helper.util.dist_proxy_config_loader import DistProxyConfigLoader

from fabric.connection import Connection

import gevent
from gevent.threadpool import ThreadPool

sys.path.insert(0, ROOT)
from test.config_loader import ConfigLoader
from test.data_splitter import DataSplitter

_DEFAULT_DP_PREFIX = '/home/ubuntu/fl-project-code/robust-secure-aggregation/scripts/'
_DEFAULT_DP_CONF = 'remote_config.ini'

_DEFAULT_PREFIX = '/home/ubuntu/fl-project-code/robust-secure-aggregation/config/'
_DEFAULT_CONF = 'lhidde_eval_mnist_l2_optim.ini'
# _DEFAULT_CONF = '/Users/hidde/IdeaProjects/robust-secure-aggregation/config/dev_config.ini'
# _DEFAULT_CONF = '/Users/hidde/IdeaProjects/fl-project-code/robust-secure-aggregation/config/lhidde_eval_mnist_l2_optim.ini'
# _DEFAULT_CONF = '/Users/hidde/IdeaProjects/fl-project-code/robust-secure-aggregation/config/lhidde_eval_cifar_lenet_l2_subspace.ini'

#_DEFAULT_CONF = '/home/mlei/Dropbox/mathias/studies/masterthesis/fed_learning/config/large_dist_nn_config.ini'


class DistProxy(object):
    
    def __init__(self, proxy_config: DistProxyConfigLoader, local_config: ConfigLoader, compile=False):
        self.dpc = proxy_config
        self.lc = local_config
        self.target_folder = self.dpc.target_folder
        # self.root = join(self.target_folder, 'fed_learning')
        self.root = self.target_folder
        self.config_dir = join(self.root, 'config')
        self.test_dir = join(self.root, 'test')
        self.log_dir = join(self.root, 'log')
        self.scripts_dir = join(self.root, 'scripts')
        self.data_dir = join(self.root, 'data')
        self.helper_dir = join(self.scripts_dir, 'helper')
        self.remote_config_path = join(self.config_dir, basename(self.lc.config_file_path))
        self.rust_dir = join(self.root, 'fed_learning', 'crypto', 'crypto_interface', 'rust_crypto')

        self.setup_script = join(self.scripts_dir, 'setup_venv.sh')
        self.bootstrap_script = join(self.helper_dir, 'bootstrap.py')
        self.test_server = join(self.test_dir, 'test_server.py')
        self.test_client = join(self.test_dir, 'test_client.py')
        self.data_splitter = join(self.test_dir, 'data_splitter.py')

        self.local_log_dir = join(LOG, 'remote')

        connect_kwargs = {"key_filename":self.dpc.key_file}        
        self.server = Connection(
            self.dpc.server,
            user=self.dpc.user,
            connect_kwargs=connect_kwargs)
        self.client_hosts = [Connection(c, user=self.dpc.user,connect_kwargs=connect_kwargs) for c in self.dpc.clients]
        self.all_hosts = [self.server] + self.client_hosts
        # number of client instances per host
        self.instances_per_client_host = [self.lc.num_clients//len(self.client_hosts) for _ in range(0, len(self.client_hosts))]
        # fill up on first host if it does not divide evenly
        self.instances_per_client_host[0] += self.lc.num_clients % len(self.client_hosts)        
        
        self.pool = ThreadPool(2*len(self.all_hosts))

    def run_cmd_all_hosts(self, cmd):
        for a in self.all_hosts:
            self.pool.spawn(a.run, cmd)
        self.pool.join()

    def sudo_cmd_all_hosts(self, cmd):
        for a in self.all_hosts:
            self.pool.spawn(a.sudo, cmd)
        self.pool.join()

    def run_all_hosts(self, cmd, *args, **kwargs) -> bool:
        fut = []
        for a in self.all_hosts:
            f = self.pool.spawn(cmd, a, *args, **kwargs)
            fut.append(f)
        self.pool.join()
        return [f.get() for f in fut]


    def shutdown_all_hosts(self):
        self.run_all_hosts(self._shutdown)

    def say_hello_all_hosts(self):
        print("all_hosts say_hello succeeded: " + str(self.run_all_hosts(self._say_hello)))

    def transfer_git_ssh_keys_all_hosts(self):
        self.run_all_hosts(self._transfer_git_ssh_key)

    def clone_repo_all_hosts(self):
        self.run_all_hosts(self._clone_repo)

    def pull_repo_all_hosts(self):
        self.run_all_hosts(self._pull_repo)

    def pip_install_hosts_all_hosts(self):
        self.run_all_hosts(self._pip_install_hosts)

    def compile_rust_all_hosts(self):
        self.run_all_hosts(self._compile_rust)

    def send_config_all_hosts(self):
        self.run_all_hosts(self._send_config)

    def send_training_data_clients(self):
        ds = DataSplitter(self.lc, shuffle=True)
        training_files = ds.split_data()
        ids = [f.split('_')[-1] for f in training_files]
        client_info_transposed = defaultdict(list)
        i=0
        for c, n in zip(self.client_hosts, self.instances_per_client_host):
            for _ in range(n):
                client_info_transposed[c].append((training_files[i], ids[i]))
                i += 1

        self.client_info = {}
        for c, info in client_info_transposed.items():
            self.client_info[c] = list(zip(*info))

        for c, info in self.client_info.items():
            self.pool.spawn(self._send_training_data, c, info[0])
        self.pool.join()

    def retrieve_logs_all_hosts(self):
        self.setup_local_log_dir()
        for c in self.client_hosts:
            self.pool.spawn(self._retrieve_logs, c)
        self.pool.spawn(self._retrieve_logs, self.server)
        self.pool.join()

    def delete_dir_all_hosts(self):
        self.run_all_hosts(self._delete_repo)

    def start_simulation(self):
        self.pool.spawn(self._start_server, self.server)
        time.sleep(3)
        print(self.client_info.items())
        for c, info in self.client_info.items():
            self.pool.spawn(self._start_client, c, info[1])
        self.pool.join()

    def clear_logs_all_hosts(self):
        self.run_all_hosts(self._clear_logs)

    def _clear_logs(self, c):
        c.run('rm -rf %s/*' % self.log_dir)

    def _start_server(self, c):
        cmd = 'AWS_DEFAULT_REGION=eu-central-1 python3 %s %s' % (self.test_server, self.remote_config_path)
        c.run(cmd)

    def _start_client(self, c, ids):
        # for i in ids:
        #     print('Starting client %s' % i)
        #     cmd = 'setsid python3 %s %s %s &' % (self.test_client, i, self.remote_config_path)
        #     c.run(cmd)
        cmd = ''
        for i in ids:
             cmd += 'AWS_DEFAULT_REGION=eu-central-1 python3 %s %s %s & ' % (self.test_client, i, self.remote_config_path)
        c.run(cmd)

    # def _set_aws_config(self, c, ids):
    #     # for i in ids:
    #     #     print('Starting client %s' % i)
    #     #     cmd = 'setsid python3 %s %s %s &' % (self.test_client, i, self.remote_config_path)
    #     #     c.run(cmd)
    #     cmd = ''
    #     for i in ids:
    #         cmd += 'AWS_DEFAULT_REGION=eu-central-1 python3 %s %s %s & ' % (self.test_client, i, self.remote_config_path)
    #     c.run(cmd)

        
    def _retrieve_logs(self, c):
        ls = c.run('ls %s' % self.log_dir)
        print("%s %s" % (c, ls))
        for f in ls.stdout.split():
            remote_log_file = join(self.log_dir, f)
            local_log_file = join(self.local_log_dir, f)
            c.get(remote_log_file, local_log_file)

    # def _retrieve_csv(self, c):
    #     c.get('~/server/')
        # for f in files:
        #     remote_log_file = join(self.log_dir, f)
        #     local_log_file = join(self.local_log_dir, f)
        #     print('retrieving %s to %s' % (remote_log_file, local_log_file))
        #     c.get(remote_log_file, local_log_file)

    def _compile_rust(self, c):
        with c.cd(self.rust_dir):
            cargo_path_var = "PATH=$PATH:~/.cargo/bin ; rustup override set nightly ; RUSTFLAGS=\"-C target_cpu=skylake-avx512\" "
            cmd = 'cargo build --release --features "fp%s frac%s"' % (self.lc.fp_bits, self.lc.fp_frac)
            c.run(cargo_path_var + cmd)

    def _send_config(self, c):
        remote_config_dir = c.run('readlink -e %s' % self.config_dir).stdout.rstrip('\n')
        c.put(self.lc.config_file_path, remote_config_dir)

    def _shutdown(self, c):
        cmd = 'shutdown -h now'
        c.run(cmd)

    def _say_hello(self, c):
        cmd = 'echo "hello from $HOSTNAME"'
        return c.run(cmd).ok

    def _transfer_git_ssh_key(self, c):
        #copy ssh config
        target_config_path='/home/ubuntu/.ssh/config'
        source_config_path='/home/ubuntu/.ssh/git_config_eval'
        c.put(source_config_path, target_config_path)

        #start ssh-agent and transfer keyfile in same shell session
        ssh_agent_cmd = 'eval "$(ssh-agent -s)"; %s'
        key_file_name=self.dpc.git_key_file.split('/')[-1]
        target_path='/home/ubuntu/.ssh/%s' % key_file_name
        install_cmd = 'ssh-add %s' % target_path
        print(self.dpc.git_key_file)
        c.put(self.dpc.git_key_file, target_path)
        c.run(ssh_agent_cmd % install_cmd)

    def _clone_repo(self, c):
        if not self._project_exists(c):
            self._transfer_git_ssh_key(c)
            # with c.cd(self.target_folder):
            c.run('git clone -v -b %s %s' % (self.dpc.branch, self.dpc.repo))

    def _pull_repo(self, c):
        if not self._project_exists(c):
            self._clone_repo(c)
            return
        with c.cd(self.target_folder):
            cmd = 'git pull origin %s' % (self.dpc.branch)
            c.run(cmd)
    
    def _pip_install_hosts(self, c):
        # cmd = 'echo $PATH'
        # cmd = 'cd %s && pip uninstall -y gevent-websocket' % (self.root)
        cmd = 'cd %s && pip install -r requirements.txt' % (self.root)
        c.run(cmd)

    def _send_training_data(self, c, files):
        if not self._file_exists(c, self.data_dir):
            with c.cd(self.root):
                c.run('mkdir %s' % self.data_dir)
        for f in files:
            print("Sending: %s" % f)
            c.put(f, self.data_dir)

    def _project_exists(self, c):
        return self._file_exists(c, self.root)

    def _delete_repo(self, c):
        c.run('rm -rf %s' % self.root)

    def _file_exists(self, c, path) -> bool:
        cmd = 'test -e %s' % path
        res = c.run(cmd, hide=False, warn=True).ok
        print(f"Result: {cmd} {res}", flush=True)
        return res

    def setup_local_log_dir(self):
        if os.path.exists(self.local_log_dir):
            cmd = ['rm', '-rf', self.local_log_dir]
            subprocess.call(cmd)
        os.mkdir(self.local_log_dir)            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Instantiate server for federated learning from config file")
    parser.add_argument('-c',  type=str, default=None, help="Custom command")
    parser.add_argument('-s',  type=str, default=None, help="Sudo command")
    parser.add_argument('-r',  action='store_true', help="Retrieve logs")
    parser.add_argument('-remote', type=str, default=_DEFAULT_DP_CONF, help="Remote config file. Default: remote_config.ini")
    parser.add_argument('-config', type=str, default=_DEFAULT_CONF, help="Config file")
    args = parser.parse_args()

    remote_config_file = _DEFAULT_DP_PREFIX + args.remote

    print(f"Loading config {args.remote} {args.config}")
    dp_config = DistProxyConfigLoader(remote_config_file)
    config = ConfigLoader(_DEFAULT_PREFIX + args.config)
    d = DistProxy(dp_config, config)

    if args.c is not None:
        print('Custom command')
        d.run_cmd_all_hosts(args.c)
        sys.exit()
    if args.s is not None:
        print('sudo command')
        d.sudo_cmd_all_hosts(args.s)
        sys.exit()
    if args.r:
        print('retrieving logs')
        d.retrieve_logs_all_hosts()
        sys.exit()

    print('preparing simulation')
    print('Server host: %s' % d.server)
    print('Client hosts: %s' % d.client_hosts)
    # print('Pulling Repo')
    # d.pull_repo_all_hosts()

    # d.delete_dir_all_hosts()

    d.run_cmd_all_hosts("killall python3")

    d.say_hello_all_hosts()
    print('sending keys')
    d.transfer_git_ssh_keys_all_hosts()
    d.clone_repo_all_hosts()
    d.pull_repo_all_hosts()
    print('sending config')
    d.send_config_all_hosts()
    print('compiling rust library')
    d.compile_rust_all_hosts()
    print('clearing logs')
    d.clear_logs_all_hosts()
    print('send sending training data in split_mode: %s' % config.split_mode)
    d.send_training_data_clients()
    # d.pip_install_hosts_all_hosts()
    print('start simulation')
    d.start_simulation()
    print('simulation finished')
    print('retrieving logs')
    d.retrieve_logs_all_hosts()
    exit()