import os
import argparse
import re

import numpy as np
import matplotlib.pyplot as plt
import json

from glob import glob
from src.tf_model import Model
from pathlib import Path


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


def get_l2_plot(benign_files, malicious_files, model_name, round):
    num_of_clients = len(benign_files) + len(malicious_files)

    benign_list_of_weights = [get_delta_weights(f, model_name) for f in benign_files]
    malicious_list_of_weights = [get_delta_weights(f, model_name) for f in malicious_files]

    benign_norms = [np.linalg.norm(b) for b in benign_list_of_weights]
    if len(malicious_list_of_weights) > 0:
        malicious_norm = np.linalg.norm(malicious_list_of_weights[0])
        return benign_norms, [malicious_norm]
    else:
        return benign_norms, []

def main():
    updates_dir, figures_dir, exp_dir = get_directories()

    files = glob(os.path.join(updates_dir, '*.npy'))
    b_mass, m_mass = [], []

    # for round in range(1, 41, 1):
    bens = []
    mals = []
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

        ben, mal = get_l2_plot(b_files, m_files, model_name, round)
        bens += ben
        mals += mal
        bensx += np.repeat(round, num_clients - num_malicious_clients).tolist()
        malsx += np.repeat(round, num_malicious_clients).tolist()

    plt.figure()
    plt.scatter(bensx, bens, label="benign")
    plt.scatter(malsx, mals, label="malicious")
    axes = plt.gca()
    axes.set_ylim([0, 0.5])
    plt.ylabel("Norm")
    plt.xlabel("Round")
    plt.title(f"Experiment {experiment_name}")
    # plt.yscale("log")
    plt.legend()
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
