import os

import h5py
import tensorflow as tf
import numpy as np


def load_data():
  path = f"{os.path.dirname(os.path.abspath(__file__))}/ARDIS_7.npy"
  (x_train, y_train), (x_test, y_test) = np.load(path, allow_pickle=True)

  # Normalize
  x_train, x_test = x_train / 255.0, x_test / 255.0

  x_train, x_test = np.moveaxis(x_train, 1, -1), np.moveaxis(x_test, 1, -1)

  return (x_train, np.argmax(y_train, axis=1)), (x_test, np.argmax(y_test, axis=1))