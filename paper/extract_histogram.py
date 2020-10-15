import ast
import argparse
import os
import re
import fnmatch
from pathlib import Path
import matplotlib.pyplot as plt


import pandas as pd
import numpy as np
import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator, HistogramEvent
from tensorboard.backend.event_processing.event_file_loader import EventFileLoader

from tensorflow.core.util.event_pb2 import Event


def extract_histogram(experiment_name, tags, wanted_steps=None):
    """
    Extract from experiment name, merges tags
    :param experiment_name:
    :param tags:
    :return: steps, actions
    """
    loader = EventFileLoader(str(experiment_name))

    # Where to store values
    wtimes, steps, actions = [], [], []
    for event in loader.Load():
        wtime = event.wall_time
        step = event.step
        if (wanted_steps is None or step in wanted_steps) and len(event.summary.value) > 0:
            summary = event.summary.value[0]
            if summary.tag in tags:
                wtimes += [wtime] * int(summary.histo.num)
                steps += [step] * int(summary.histo.num)

                for num, val in zip(summary.histo.bucket, summary.histo.bucket_limit):
                    actions += [val] * int(num)

    # print(actions)
    actions = np.array(actions)
    steps = np.array(steps)

    # norm = np.linalg.norm(actions)
    # print(norm)
    # this_round = actions[steps == 80]
    #
    # second_round = actions[steps == 160]
    # print(this_round, (steps == 160))
    # plt.figure()
    # plt.hist(this_round)
    # plt.hist(second_round, color='#FF00FF')
    # # plt.plot(steps, actions)
    # plt.show()
    return steps, actions


if __name__ == '__main__':
    
    def param_list(param):
        p_list = ast.literal_eval(param)
        if type(p_list) is not list:
            raise argparse.ArgumentTypeError("Parameter {} is not a list".format(param))
        return p_list

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="main path for tensorboard files", default=os.getcwd())
    parser.add_argument("--subpaths", type=param_list, help="subpath structures", default=['test', 'train'])
    parser.add_argument("--output", type=str, help="aggregation can be saved as tensorboard file (summary) or as table (csv)", default='summary')
    parser.add_argument("--experiments", type=str, help="experiments", default=[], nargs='+')
    parser.add_argument("--tags", type=str, help="Keys of the data points to export", default=["evaluation/test_accuracy", "evaluation/adv_success"], nargs='+')

    args = parser.parse_args()

    path = Path(args.path)

    if not path.exists():
        raise argparse.ArgumentTypeError("Parameter {} is not a valid path".format(path))

    # subpaths = [path / dname / subpath for subpath in args.subpaths for dname in os.listdir(path) if dname != FOLDER_NAME and os.path.isdir(path / dname) and dname in args.experiments]
    #
    # for subpath in subpaths:
    #     if not os.path.exists(subpath):
    #         raise argparse.ArgumentTypeError("Parameter {} is not a valid path".format(subpath))
    #
    # if args.output not in ['summary', 'csv']:
    #     raise argparse.ArgumentTypeError("Parameter {} is not summary or csv".format(args.output))

    #extract_histogram(path, args.tags, wanted_steps=None)

