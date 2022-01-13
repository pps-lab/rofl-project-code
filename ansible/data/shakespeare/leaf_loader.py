

"""Loads leaf datasets"""
import os
import numpy as np
import pathlib

from leaf.model_utils import read_data


def load_leaf_dataset(dataset, use_val_set=False):

    eval_set = 'test' if not use_val_set else 'val'

    base_dir = pathlib.Path(__file__).parent.resolve()

    train_data_dir = os.path.join(base_dir, 'leaf', dataset, 'data', 'train')
    test_data_dir = os.path.join(base_dir, 'leaf', dataset, 'data', eval_set)

    users, groups, train_data, test_data = read_data(train_data_dir, test_data_dir)

    return users, train_data, test_data


# ------------------------
# utils for shakespeare dataset

ALL_LETTERS = "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
NUM_LETTERS = len(ALL_LETTERS)


def word_to_indices(word):
    '''returns a list of character indices

    Args:
        word: string

    Return:
        indices: int list with length len(word)
    '''
    indices = []
    for c in word:
        indices.append(ALL_LETTERS.find(c))
    return indices


def _one_hot(index, size):
    '''returns one-hot vector with given size and value 1 at given index
    '''
    vec = [0 for _ in range(size)]
    vec[int(index)] = 1
    return vec


def letter_to_vec(letter):
    '''returns one-hot representation of given letter
    '''
    index = ALL_LETTERS.find(letter)
    return index
    # return _one_hot(index, NUM_LETTERS)


def process_text_input_indices(x_batch: list):
    x_batch = [word_to_indices(word) for word in x_batch]
    x_batch = np.array(x_batch)
    return x_batch


def process_char_output_indices(y_batch: list):
    y_batch = [letter_to_vec(c) for c in y_batch]
    y_batch = np.array(y_batch, dtype=np.uint8)
    return y_batch
