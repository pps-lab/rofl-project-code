from os.path import join, exists, dirname, abspath
import sys
import subprocess

SCRIPTS = dirname(dirname(dirname(abspath(__file__))))
ROOT = abspath(dirname(SCRIPTS))

CONFIG = join(ROOT, 'config')
TEST = join(ROOT, 'test')
SRC = join(ROOT, 'fed_learning')
LOG = join(ROOT, 'log')
CRYPTO = join(SRC, 'crypto')
RUST_CRYPTO = join(CRYPTO, 'crypto_interface', 'rust_crypto')
HELPER = join(SCRIPTS, 'helper')

OLD_CONFIG_FILE = join(CONFIG, '.old.ini')
DATASPLITTER = join(TEST, 'data_splitter.py')
LOCAL_HELPER_SCRIPT = join(HELPER, 'local_sim_helper.py')
LOCAL_HELPER_SCRIPT_SH = join(HELPER, 'local_sim_helper_sh.py')
REMOTE_HELPER_SCRIPT = join(HELPER, 'remote_sim_helper.py')
REMOTE_SERVER_SCRIPT = join(HELPER, 'remote_server.py')
REMOTE_CLIENT_SCRIPT = join(HELPER, 'remote_client.py')

sys.path.insert(0, ROOT)
from test.config_loader import ConfigLoader

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def crypto_config_changed(config: ConfigLoader, old_config: ConfigLoader):
    return (not old_config.fp_bits == config.fp_bits) or \
            (not old_config.fp_frac == config.fp_frac) or \
            (not old_config.training_size == config.training_size) or \
            (not old_config.test_size == config.test_size)

def dataset_config_changed(config: ConfigLoader, old_config: ConfigLoader):
    return (not old_config.num_clients == config.num_clients) or \
        (not old_config.dataset == config.dataset) or \
        (not old_config.test_size == config.test_size) or \
        (not old_config.training_size == config.training_size)

def run_data_splitter(config: ConfigLoader):
    logger.info('Redistributing %s datasets' % config.dataset)
    subprocess.call(['python3', DATASPLITTER, config.config_file_path, '-s'])

def compile_crypto(config: ConfigLoader):
    fp_bits = 'fp' + str(config.fp_bits)
    fp_frac = 'frac' + str(config.fp_frac)
    flags = "%s %s" % (fp_bits, fp_frac)
    logger.info('Compiling rust crypto library with flags: %s' % flags)
    cmd = ['cargo', 'build', '--release', '--features', flags]
    subprocess.call(cmd, cwd=RUST_CRYPTO)

def copy_to_old_config(config: ConfigLoader):
    cmd = ['cp', config.config_file_path, OLD_CONFIG_FILE]
    subprocess.call(cmd)

def bootstrap(config: ConfigLoader, compile=False, distribute_data=False):
    if not exists(OLD_CONFIG_FILE):
        logger.info('No previous simulation detected -> setting up fresh environment')
        run_data_splitter(config)
        compile_crypto(config)
        copy_to_old_config(config)
        return

    old_config = ConfigLoader(OLD_CONFIG_FILE, log_summary=False)
    
    if crypto_config_changed(config, old_config) or compile:
        logger.info("Crypto configurations changed")
        compile_crypto(config)

    if dataset_config_changed(config, old_config) or distribute_data:
        logger.info("Dataset configurations changed")
        run_data_splitter(config)

    copy_to_old_config(config)
    return
