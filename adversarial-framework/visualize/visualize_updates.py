import argparse
import os
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.tf_model import Model
from visualize.utils import split_mal_ben_client_files


def get_directories(experiment_name):
    updates_dir = os.path.join('.', 'experiments', experiment_name, 'updates')
    figures_dir = os.path.join('.', 'experiments', experiment_name, 'figures')

    if not os.path.isdir(figures_dir):
        os.mkdir(figures_dir)
    else:
        for filename in Path(figures_dir).glob('**/*'):
            if not os.path.isdir(filename):
                os.remove(filename)

    return updates_dir, figures_dir


def get_delta_weights(pickle_file):
    to_ret = []
    # with open(pickle_file, 'rb') as handle:
    weights = np.load(pickle_file, allow_pickle=True)  # these are model updates
    # weights = pickle.load(handle)
    model = Model.create_model(ARGS.model_name)
    model.set_weights(weights)
    for i in range(len(model.layers)):
        w = model.layers[i].get_weights()
        if w:
            w = w[0].flatten()  # 0 is weights, 1 is biases
            to_ret = to_ret + w.tolist()

    return np.array(to_ret)


def average_updates(list_of_weights, num_of_clients):
    alpha = 1. / num_of_clients

    w = alpha * np.array(list_of_weights[0])
    for i in range(1, len(list_of_weights)):
        w += alpha * list_of_weights[i]

    return w


def get_histogram(benign_files, malicious_files, round):
    num_of_clients = len(benign_files) + len(malicious_files)

    benign_list_of_weights = [get_delta_weights(f) for f in benign_files]
    malicious_list_of_weights = [get_delta_weights(f) for f in malicious_files]

    # benign_weights = average_updates(benign_list_of_weights, num_of_clients)
    benign_weights = np.array([x.tolist() for x in benign_list_of_weights]).flatten().tolist()
    # malicious_weights = average_updates(malicious_list_of_weights, num_of_clients)
    malicious_weights = malicious_list_of_weights[0]

    norm = np.linalg.norm(benign_weights)
    mal_norm = np.linalg.norm(malicious_weights)
    print(norm, mal_norm)

    # benign_weights = benign_weights[np.where(np.abs(benign_weights) > 1e-7)]
    # malicious_weights = malicious_weights[np.where(np.abs(malicious_weights) > 1e-7)]

    b_hist, _ = np.histogram(benign_weights, bins=BINS)
    m_hist, _ = np.histogram(malicious_weights, bins=BINS)

    b_hist, m_hist = b_hist / b_hist.sum(), m_hist / m_hist.sum()

    plt.bar(BINS[:-1], b_hist, align="edge", width=np.diff(BINS), label='Benign')
    plt.bar(BINS[:-1], m_hist, align="edge", width=np.diff(BINS), color='r', label='Malicious')

    # plt.ylim((0, Y_LIMIT))
    plt.xlabel('Local model updates')
    plt.legend()

    return b_hist, m_hist


def append_mass_to_log(b_mass, m_mass):
    import pandas as pd
    log_file = os.path.join('.', 'experiments', ARGS.experiment_name, 'log.csv')
    if os.path.exists(log_file):
        df = pd.read_csv(log_file)
        df['b_mass'] = b_mass
        df['m_mass'] = m_mass
        df.to_csv(log_file, index=False)
    else:
        print('log file does not exist')


def main():
    updates_dir, figures_dir = get_directories(ARGS.experiment_name)

    files = glob(os.path.join(updates_dir, '*.npy'))
    b_mass, m_mass = [], []

    # for round in range(1, 41, 1):
    for round in range(23, 28, 1):
        b_files, m_files = split_mal_ben_client_files(files, round)

        plt.figure()
        b_hist, m_hist = get_histogram(b_files, m_files, round)
        plt.savefig(os.path.join(figures_dir, 'hist_%i.pdf' % round), format='pdf')
        plt.savefig(os.path.join(figures_dir, 'hist_%i.png' % round), format='png')
        if ARGS.clip:
            MASS_INDS = np.where(np.abs(BINS) > ARGS.clip)[0][:-1]
            b_mass, m_mass = b_mass + [b_hist[MASS_INDS].sum()], m_mass + [m_hist[MASS_INDS].sum()]
            title = "Round %i MASSb=%f MASSm=%f" % (round, b_mass[-1], m_mass[-1])
        else:
            title = "Round %i" % round

        # plt.title(title)
        # plt.show()
        print(title)

    if ARGS.clip:
        append_mass_to_log(b_mass, m_mass)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--experiment_name", type=str,
                        default='targeted_mal_baseline_boosting_true_num_malicious_clients_1_mnist_clients_25',
                        help="Log file path.")
    parser.add_argument("--model_name", type=str, default='mnist_cnn', help="Which model to use.",
                        choices=['dev', 'mnist_cnn'])
    parser.add_argument("--clip", type=float, default=None, help="A positive value for absolute update clipping.")
    # Y_LIMIT = 22500

    Y_LIMIT = .15

    ARGS = parser.parse_args()

    # BINS = np.linspace(-.05, .05, 200)  # good limit for 10
    # BINS = np.linspace(-0.15, 0.15, 200)  # good limit for 25
    BINS = np.linspace(-0.002, 0.002, 200)
    main()
