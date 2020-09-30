import os
import argparse
import re
from os.path import join

import numpy as np
import matplotlib.pyplot as plt
import json

from glob import glob

import pandas as pd

from plotting.baseline import parse_log
from src.tf_model import Model
from pathlib import Path

def get_accuracy(experiment_name):
    dfs = get_df_list("", experiment_name)
    obj = dfs[0]
    data = obj["df"].sort_values(by=['round'])
    accuracy, rounds = data['adv_success'].values, data['round'].values
    return accuracy, rounds

def get_df_list(directory_wildcard='hist*', experiment_name = ""):
    log_dir = join('.', 'experiments')
    dirs = []

    for n in experiment_name.split(","):
        dirs.extend(glob(join(log_dir, n)))

    dfs = []
    for exp_dir in dirs:
        files = glob(join(exp_dir, 'log.csv'))
        assert len(files) > 0, f"{exp_dir} not well-formatted"
        accuracy, rounds, adv_success = parse_log(files[0])
        config_dict = parse_config(glob(join(exp_dir, 'config.json'))[0])

        table_values = {
            'dataset': [],
            'num_clients': [],
            'number_of_samples': [],
            'attack_type': [],
            'scale_attack': [],
            'scale_attack_weight': [],
            'num_malicious_clients': [],

            'accuracy': [],
            'adv_success': [],
            'round': [],
        }
        table_values['accuracy'] += accuracy
        table_values['adv_success'] += adv_success
        table_values['round'] += rounds
        for key in ['dataset', 'num_clients', 'number_of_samples', 'attack_type', 'scale_attack', 'scale_attack_weight', 'num_malicious_clients']:
            table_values[key] = table_values[key] + [config_dict[key]] * len(accuracy)

        df = pd.DataFrame(table_values)
        obj = {
            'df': df,
            'config': config_dict
        }
        dfs.append(obj)

    return dfs


def get_directories():
    updates_dir = os.path.join('.', 'experiments', ARGS.experiment_name, 'updates')
    figures_dir = os.path.join('.', 'experiments', ARGS.experiment_name, 'figures')
    exp_dir = os.path.join('.', 'experiments', ARGS.experiment_name)

    if not os.path.isdir(figures_dir):
        os.mkdir(figures_dir)
    else:
        for filename in Path(figures_dir).glob('**/*'):
            if not os.path.isdir(filename):
                os.remove(filename)

    return updates_dir, figures_dir, exp_dir


def get_delta_weights(pickle_file, model_name):
    to_ret = []
    # with open(pickle_file, 'rb') as handle:
    weights = np.load(pickle_file, allow_pickle=True)  # these are model updates
    # weights = pickle.load(handle)
    model = Model.create_model(model_name)
    model.set_weights(weights)
    for i in range(len(model.layers)):
        w = model.layers[i].get_weights()
        if w:
            w = w[0].flatten()  # 0 is weights, 1 is biases
            to_ret = to_ret + w.tolist()

    return np.array(to_ret)

def parse_config(config_file):
    with open(config_file) as f:
        json_data = f.read()
        config_dict = (json.loads(json_data))
        return config_dict


def get_n0_vals(benign_files, malicious_files, model_name, round):
    num_of_clients = len(benign_files) + len(malicious_files)

    benign_list_of_weights = [get_delta_weights(f, model_name) for f in benign_files]
    malicious_list_of_weights = [get_delta_weights(f, model_name) for f in malicious_files]

    benign_data = [{'mean': b.mean(), 'min': b.min(), 'max': b.max()} for b in benign_list_of_weights]
    malicious_data = [{'mean': b.mean(), 'min': b.min(), 'max': b.max()} for b in malicious_list_of_weights]

    return benign_data, malicious_data

def main():
    updates_dir, figures_dir, exp_dir = get_directories()

    files = glob(os.path.join(updates_dir, '*.npy'))
    b_mass, m_mass = [], []

    # for round in range(1, 41, 1):
    bens = []
    mals = []
    benerr = []
    malerr = []
    bensx = []
    malsx = []
    config_dict = parse_config(glob(os.path.join(exp_dir, 'config.json'))[0])
    num_clients = config_dict["num_clients"]
    num_malicious_clients = config_dict["num_malicious_clients"]
    model_name = config_dict["model_name"]
    experiment_name = config_dict["experiment_name"]

    for round in range(1, config_dict["num_rounds"], 1):
        print(f"Round {round}")
        b_files = [file for file in files if re.search('_b_%i.npy' % round, file)]
        m_files = [file for file in files if re.search('_m_%i.npy' % round, file)]

        ben, mal = get_n0_vals(b_files, m_files, model_name, round)
        bens += [b["mean"] for b in ben]
        mals += [b["mean"] for b in mal]
        benerr += [[abs(b["min"] - b["mean"]), abs(b["max"] - b["mean"])] for b in ben]
        malerr += [[abs(b["min"] - b["mean"]), abs(b["max"] - b["mean"])] for b in mal]
        bensx += np.repeat(round, len(ben)).tolist()
        malsx += np.repeat(round, num_malicious_clients).tolist()

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.errorbar(bensx, bens, np.array(benerr).T, label = "benign")
    ax1.errorbar(malsx, mals, np.array(malerr).T, label= "malicious", c=(1.0, 0.0, 0.0, 0.25))
    ax1.set_ylabel("Weight value")
    ax1.set_xlabel("Round")
    plt.title(f"Experiment {experiment_name}")

    accuracy, rounds = get_accuracy(experiment_name)
    ax2.plot(rounds, accuracy, label=f'Malicious objective', c=(0.0, 1.0, 0.0, 0.25))
    ax2.set_ylim([0, 1.2])
    ax2.set_ylabel("Malicious objective")
    ax2.spines['right'].set_color((0.0, 1.0, 0.0, 0.4))
    # plt.yscale("log")

    ax1.legend()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--experiment_name", type=str, default='targeted_mal_baseline_boosting_true_num_malicious_clients_1_mnist_clients_25', help="Log file path.")
    parser.add_argument("--clip", type=float, default=None, help="A positive value for absolute update clipping.")
    # Y_LIMIT = 22500

    Y_LIMIT = .15

    ARGS = parser.parse_args()

    # BINS = np.linspace(-.05, .05, 200)  # good limit for 10
    # BINS = np.linspace(-0.15, 0.15, 200)  # good limit for 25
    BINS = np.linspace(-0.002, 0.002, 200)
    main()
