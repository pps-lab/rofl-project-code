import os
import sys
module_loc = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(module_loc))


import shutil
import pickle
import random
from os.path import join, abspath, dirname, isabs
from importlib import import_module
import argparse
import numpy as np

from test.config_loader import ConfigLoader

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ROOT = dirname(dirname(abspath(__file__)))
DATA = join(ROOT, 'data')

class DataSplitter(object):
    
    # mode 
    def __init__(self, config: ConfigLoader, shuffle=False, **kwargs):

        self.num_clients = config.num_clients
        self.dataset = config.dataset
        self.shuffle = shuffle
        self.training_size = config.training_size
        self.test_size = config.test_size
        self.kwargs = kwargs
        self.module_name = 'keras.datasets.' + self.dataset
        self.data_module = import_module(self.module_name)
        self.mode = config.split_mode

    def split_data(self):
        logger.info('Splitting mode: %s' % self.mode)
        if self.mode == 'fixed':
            return self.split_data_fixed()
        return self.split_data_even()

    def setup_data_dir(self):
        if not os.path.exists(DATA):
            os.mkdir(DATA)
        else:
            shutil.rmtree(DATA)
            os.mkdir(DATA)
        
    def split_data_fixed(self):
        self.setup_data_dir()

        logger.info('Loading %s dataset', self.dataset)
        file_prefix = self.dataset + '_'
        (x_train, y_train), (x_test, y_test) = self.data_module.load_data(**self.kwargs)
        # x_train, x_test = x_train / 255.0, x_test / 255.0

        logger.info('Loaded %d training samples from %s', len(x_train), self.module_name)
        logger.info('Loaded %d test samples from %s', len(x_test), self.module_name)
        logger.info('Splitting %s dataset into %d files', self.dataset, self.num_clients)

        test_data_size = len(x_test) // self.num_clients
        if self.test_size is not None:
            test_data_size = min(test_data_size, self.test_size)
        logger.info('Limit test sample size to %d', test_data_size)

        training_data = list(zip(x_train, y_train))
        file_paths = []
        for i in range(0, self.num_clients):
            tr_data = random.sample(training_data, self.training_size)
            x_tr, y_tr = list(zip(*tr_data))

            x_te = x_test[:test_data_size]
            y_te = y_test[:test_data_size]

            data = (x_tr, y_tr, x_te, y_te)

            file_path = os.path.join(DATA, file_prefix + str(i+1))
            fh = open(file_path, 'wb')
            file_paths.append(file_path)
            pickle.dump(data, fh)
            fh.close()
        return file_paths


    def split_data_even(self):
        self.setup_data_dir()

        logger.info('Loading %s dataset', self.dataset)
        file_prefix = self.dataset + '_'
        (x_train, y_train), (x_test, y_test) = self.data_module.load_data(**self.kwargs)

        if self.shuffle:
            logger.info('Shuffling data')
            train_data = list(zip(x_train, y_train))
            #test_data = list(zip(x_test, y_test))

            random.shuffle(train_data)
            #random.shuffle(test_data)

            train_data = list(zip(*train_data))
            #test_data = list(zip(*test_data))

            x_train = list(train_data[0])
            y_train = list(train_data[1])
            #x_test = list(test_data[0])
            #y_test = list(test_data[1])

        logger.info('Loaded %d training samples from %s', len(x_train), self.module_name)
        logger.info('Loaded %d test samples from %s', len(x_test), self.module_name)
        logger.info('Splitting %s dataset into %d files', self.dataset, self.num_clients)
    
        train_data_size = len(x_train) // self.num_clients
        if self.training_size is not None:
            train_data_size = min(train_data_size, self.training_size)
        logger.info('Limit training sample size to %d', train_data_size)

        test_data_size = len(x_test) // self.num_clients
        if self.test_size is not None:
            test_data_size = min(test_data_size, self.test_size)
        logger.info('Limit test sample size to %d', test_data_size)

        file_paths = []
        for i in range(0, self.num_clients):
            idx_tr = i*train_data_size
            x_tr = x_train[idx_tr:idx_tr+train_data_size]
            y_tr = y_train[idx_tr:idx_tr+train_data_size]

            idx_te = i*test_data_size
            x_te = x_test[idx_te:idx_te+test_data_size]
            y_te = y_test[idx_te:idx_te+test_data_size]

            data = (x_tr, y_tr, x_te, y_te)
            file_path = os.path.join(DATA, file_prefix + str(i+1))
            fh = open(file_path, 'wb')
            file_paths.append(file_path)
            pickle.dump(data, fh)
            fh.close()
        return file_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='split specified into equally sized chunks')
    parser.add_argument('config_file_path', type=str, help="Configuration file location")
    parser.add_argument('-s', '--shuffle', action='store_true', help='shuffle data')
    # parser.add_argument('dataset', type=str, help='dataset to be split from the keras library (https://keras.io/datasets/)')
    # parser.add_argument('num_files', type=int, help='number of files to split into')
    # parser.add_argument('-a', '--training_size', type=int, help='Limit the number of training samples used (defaults to complete test file split')
    # parser.add_argument('-e', '--test_size', type=int, help='Limit the number of test samples used (defaults to complete test file split')
    args = parser.parse_args()
    
    config = ConfigLoader(args.config_file_path)
    logging.getLogger(__name__).addHandler(logging.StreamHandler())

    loader = DataSplitter(config,
                          shuffle=args.shuffle,
)
    loader.split_data()

