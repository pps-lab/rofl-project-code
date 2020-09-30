import os
import pickle

import numpy as np

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DataLoader(object):

    def __init__(self, data_path, file_prefix, client_id, transform=None):
        self.client_id = client_id
        self.data_path = data_path
        self.file_prefix = file_prefix
        self.transform = transform
        
    def load_data(self):
        file_path = os.path.join(self.data_path, self.file_prefix + str(self.client_id))
        logger.info('Loading data from ' + file_path)
        print('Loading data from ' + file_path)
        if not os.path.exists(file_path):
            print(f"{file_path} file not found!")
            logger.error(f"{file_path} file not found!")
            exit(0)

        with open(file_path, 'rb') as fh:
            x_train, y_train, x_test, y_test = pickle.load(fh)

        logger.info('Loaded %d training samples', len(x_train))
        logger.info('Loaded %d test samples', len(x_test))

        print('Loaded %d training samples', len(x_train))
        print('Loaded %d test samples', len(x_test))

        if self.transform is not None:
            return self.transform(x_train, y_train, x_test, y_test)

        print("Data load done")

        return (x_train / 255.0, y_train), (x_test / 255.0, y_test)
