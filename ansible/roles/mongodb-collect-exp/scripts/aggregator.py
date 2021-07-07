
import ast
import argparse
import os
import re
import fnmatch
from pathlib import Path

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorflow.core.util.event_pb2 import Event

FOLDER_NAME = 'aggregates'

# keys = ["evaluation/test_accuracy", "evaluation/adv_success"]

# keys = ["evaluation/test_accuracy", "evaluation/adv_success", "evaluation/honest_train_accuracy", "adversary/count"]
# keys_add = [0, 0, 1, 1]  # Add to make lists same size

def extract(dpath, subpath, experiments, keys):
    if len(experiments) > 1:
        scalar_accumulators = {dname: EventAccumulator(str(dpath + '/' + dname + '/' + subpath)).Reload(
        ).scalars for dname in os.listdir(dpath) if dname != FOLDER_NAME and dname in experiments}
    elif len(experiments) == 1:
        # Pattern match
        scalar_accumulators = {dname: EventAccumulator(str(dpath + '/' + dname + '/' + subpath)).Reload(
        ).scalars for dname in os.listdir(dpath) if dname != FOLDER_NAME and len(fnmatch.filter([dname], experiments[0])) > 0}
    else:
        scalar_accumulators = {'0': EventAccumulator(str(dpath + '/' + subpath)).Reload(
        ).scalars}

    # Filter non event files
    scalar_accumulators = {l: scalar_accumulator for (l, scalar_accumulator) in scalar_accumulators.items() if scalar_accumulator.Keys()}

    # Get and validate all scalar keys
    # all_keys
    # all_keys = [tuple(scalar_accumulator.Keys()) for scalar_accumulator in scalar_accumulators]
    # assert len(set(all_keys)) == 1, "All runs need to have the same scalar keys. There are mismatches in {}".format(all_keys)
    # keys = all_keys[0]

    all_scalar_events_per_run = {l: {key: scalar_accumulator.Items(key) for key in keys} for (l, scalar_accumulator) in scalar_accumulators.items()}

    return all_scalar_events_per_run


def aggregate_to_summary(dpath, aggregation_ops, extracts_per_subpath):
    for op in aggregation_ops:
        for subpath, all_per_key in extracts_per_subpath.items():
            path = dpath / FOLDER_NAME / op.__name__ / dpath.name / subpath
            aggregations_per_key = {key: (steps, wall_times, op(values, axis=0)) for key, (steps, wall_times, values) in all_per_key.items()}
            write_summary(path, aggregations_per_key)


def write_summary(dpath, aggregations_per_key):
    writer = tf.summary.FileWriter(dpath)

    for key, (steps, wall_times, aggregations) in aggregations_per_key.items():
        for step, wall_time, aggregation in zip(steps, wall_times, aggregations):
            summary = tf.Summary(value=[tf.Summary.Value(tag=key, simple_value=aggregation)])
            scalar_event = Event(wall_time=wall_time, step=step, summary=summary)
            writer.add_event(scalar_event)

        writer.flush()


def aggregate_to_csv(dpath, extracts_per_subpath):
    flattened = {}
    runs = ""
    for subpath, rundict in extracts_per_subpath.items():
        for run, all_per_key in rundict.items():
            runs += run
            for key, vals in all_per_key.items():
                flattened[f"{run}_{key}"] = vals

    vals = [[v.value for v in events] for (_, events) in flattened.items()]
    max_vals = max([len(v) for v in vals])
    steps = range(1, max_vals + 1)

    for i, v in enumerate(vals):
        add = max_vals - len(v)
        for _ in range(add):
            v.append(float('nan'))
    vals = np.array(vals)

    path = dpath / FOLDER_NAME

    if not path.exists():
        os.makedirs(path)

    file_name = get_valid_filename(runs) + '-' + "output" + '.csv'
    aggregation_ops_names = [aggregation_op for aggregation_op in flattened.keys()]
    df = pd.DataFrame(np.transpose(vals), index=steps, columns=aggregation_ops_names)
    df.to_csv(path / file_name, sep=',')


def aggregate_to_df(dpath, extracts_per_subpath):
    flattened = {}
    runs = ""
    for subpath, rundict in extracts_per_subpath.items():
        for run, all_per_key in rundict.items():
            runs += run
            for key, vals in all_per_key.items():
                flattened[f"{run}_{key}"] = vals

    vals = [[v.value for v in events] for (_, events) in flattened.items()]
    max_vals = max([len(v) for v in vals])
    steps = range(1, max_vals + 1)

    for i, v in enumerate(vals):
        add = max_vals - len(v)
        for _ in range(add):
            v.append(float('nan'))
    vals = np.array(vals)

    aggregation_ops_names = [aggregation_op for aggregation_op in flattened.keys()]
    df = pd.DataFrame(np.transpose(vals), index=steps, columns=aggregation_ops_names)
    return df


def get_valid_filename(s):
    s = str(s).strip().replace(' ', '_')
    if len(s) > 100:
        import time
        t = time.time()
        s = s[0:100] + '_' + str(t)
    return re.sub(r'(?u)[^-\w.]', '', s)


def write_csv(dpath, fname, aggregations, steps, aggregation_ops):
    path = dpath / FOLDER_NAME

    if not path.exists():
        os.makedirs(path)

    file_name = get_valid_filename(subpath) + '-' + fname + '.csv'
    aggregation_ops_names = [aggregation_op.__name__ for aggregation_op in aggregation_ops]
    df = pd.DataFrame(np.transpose(aggregations), index=steps, columns=aggregation_ops_names)
    df.to_csv(path / file_name, sep=',')


def aggregate(dpath, output, subpaths, experiments, keys):
    # name = dpath.name

    ops = {
        'summary': aggregate_to_summary,
        'csv': aggregate_to_csv,
        'df': aggregate_to_df
    }

    # print(dpath)
    # print(subpaths)
    # print(experiments)

    # print("Started aggregation {}".format(dpath))

    extracts_per_subpath = {subpath: extract(dpath, subpath, experiments, keys) for subpath in subpaths}

    return ops.get(output)(dpath, extracts_per_subpath)


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

    subpaths = [path / dname / subpath for subpath in args.subpaths for dname in os.listdir(path) if dname != FOLDER_NAME and os.path.isdir(path / dname) and dname in args.experiments]

    for subpath in subpaths:
        if not os.path.exists(subpath):
            raise argparse.ArgumentTypeError("Parameter {} is not a valid path".format(subpath))

    if args.output not in ['summary', 'csv']:
        raise argparse.ArgumentTypeError("Parameter {} is not summary or csv".format(args.output))

    aggregate(path, args.output, args.subpaths, args.experiments, args.tags)

    # print("Ended aggregation {}".format(path.name))