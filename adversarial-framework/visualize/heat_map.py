import os
import argparse
import re
import numpy as np
import matplotlib.pyplot as plt

from glob import glob
from src.tf_model import Model
from pathlib import Path
from copy import deepcopy


def get_directories():
    """Creates and/or cleans directory in plots will be saved.

    Notes:
        This function cannot locate the experiment directory if this script is ran as module.
    """
    updates_dir = os.path.join('.', 'experiments', ARGS.experiment_name, 'updates')
    figures_dir = os.path.join('.', 'experiments', ARGS.experiment_name, 'heat_maps')

    if not os.path.isdir(figures_dir):
        os.mkdir(figures_dir)
    else:
        for filename in Path(figures_dir).glob('**/*'):
            if not os.path.isdir(str(filename)):
                os.remove(str(filename))

    return updates_dir, figures_dir


def get_delta_weights(pickle_file):
    """Loads updates."""
    to_ret = []
    weights = np.load(pickle_file, allow_pickle=True)  # local updates
    # with open(pickle_file, 'rb') as handle:
    model = Model.create_model(ARGS.model_name)
    model.set_weights(weights)
    for i in range(len(model.layers)):
        w = model.layers[i].get_weights()
        if w:
            w = w[0].flatten()  # 0 is weights, 1 is biases
            to_ret.append(w)

    return to_ret


def sum_weights(list_of_weights):
    """Sums provided updates."""
    weights = deepcopy(list_of_weights[0])

    num_of_layers = len(weights)
    for delta_weights in list_of_weights[1:]:
        for l in range(num_of_layers):
            weights[l] = weights[l] + delta_weights[l]

    return weights


def create_plot(bw, mw, round, layer, cmap='gray'):
    """Produces plots.

    Args:
        bw: benign updates
        mw: malicious updates
        round (int): updates from which round
        layer (int): which layers is used
        cmap (str): color map for matplotlib
    """
    min_max = -0.003, 0.003
    # min_max = min(bw.min(), mw.min()), max(bw.max(), mw.max())
    plt.figure()
    plt.subplot(121)
    plt.imshow(bw, cmap=cmap, interpolation='nearest', vmin=min_max[0], vmax=min_max[1])
    plt.colorbar(shrink=.45)
    # plt.title('B Round=%s, layer=%s' % (round, layer + 1))
    plt.title('Benign updates')

    plt.subplot(122)
    plt.imshow(mw, cmap=cmap, interpolation='nearest', vmin=min_max[0], vmax=min_max[1])
    plt.colorbar(shrink=.45)
    # plt.title('M Round=%s, layer=%s' % (round, layer + 1))
    plt.title('Malicious updates')

    plt.tight_layout()


def generate_delta_weights(layer, benign_list_of_weights, malicious_list_of_weights, sum_ben_updates):
    """Prepares updates for plotting.

    Warnings: Currently we assume that there is only one malicious client and that its updates are scaled by the
    factor of the total number of clients.

    Args:
        layer (int): which layer to use
        benign_list_of_weights (list): list of benign weights
        malicious_list_of_weights (list): list of malicious weights
        sum_ben_updates (bool): Which strategy to use. If set to True, benign updates will be summed up, otherwise the
    malicious updates will be scaled down by the factor of the number of clients and benign updates would be provided
    from a randomly selected honest client.
    """
    if layer != 0:
        raise Exception("Only first layer supported")

    benign_list_of_weights = deepcopy(benign_list_of_weights)
    malicious_list_of_weights = deepcopy(malicious_list_of_weights)
    if sum_ben_updates:
        ben_weights = sum_weights(benign_list_of_weights)
    else:
        ben_weights = benign_list_of_weights[0]  # randomly select updates from 1 user

    mal_weights = malicious_list_of_weights[0]  # compute benign weights

    new_size = int(np.sqrt(ben_weights[layer].shape[0]))
    bw = ben_weights[layer].reshape((new_size, new_size))
    mw = mal_weights[layer].reshape((new_size, new_size))

    if not sum_ben_updates:  # if ben updates are not, then mal ones should be scaled down
        num_of_clients = len(benign_list_of_weights) + len(malicious_list_of_weights)
        mw /= num_of_clients  # scale back updates

    # normalization since we are interested only in correlation
    bw = bw/bw.sum()
    mw = mw/mw.sum()

    return bw, mw


def generate_correlations(layer, benign_list_of_weights, malicious_list_of_weights, sum_ben_updates):
    """Generates correlation matrix. Arguments are the same as for `generate_delta_weights` function."""
    benign_list_of_weights = deepcopy(benign_list_of_weights)  # not the most efficient way to solve a bug
    malicious_list_of_weights = deepcopy(malicious_list_of_weights)

    if sum_ben_updates:
        ben_weights = sum_weights(benign_list_of_weights)
    else:
        ben_weights = benign_list_of_weights[0]  # randomly select updates from 1 user

    mal_weights = malicious_list_of_weights[0]  # compute benign weights
    import pandas as pd
    print('MAL_W\n', pd.Series(np.abs(mal_weights[0])).describe())
    print('BEN_W\n', pd.Series(np.abs(ben_weights[0])).describe())
    bw = np.outer(ben_weights[layer], ben_weights[layer])
    mw = np.outer(mal_weights[layer], mal_weights[layer])

    if sum_ben_updates is False:  # if ben updates are not, then mal ones should be scaled down
        num_of_clients = len(benign_list_of_weights) + len(malicious_list_of_weights)
        mw /= num_of_clients  # scale back updates

    # normalization since we are interested only in correlation
    bw = bw/bw.sum()
    mw = mw/mw.sum()

    return bw, mw


def main():
    updates_dir, figures_dir = get_directories()

    files = glob(os.path.join(updates_dir, '*.npy'))

    # for round in range(1, 41, 1):
    for round in range(23, 28, 1):
        print(round)
        b_files = [file for file in files if re.search('_b_%i.npy' % round, file)]
        m_files = [file for file in files if re.search('_m_%i.npy' % round, file)]

        benign_list_of_weights = [get_delta_weights(f) for f in b_files]
        malicious_list_of_weights = [get_delta_weights(f) for f in m_files]

        for i in range(0, len(benign_list_of_weights[0])):
            # gradients
            # bw, mw = generate_delta_weights(i, benign_list_of_weights, malicious_list_of_weights, False)
            # create_plot(bw, mw, round, i)
            # plt.savefig(os.path.join(figures_dir, 'rnd updates - round %s_layer %s.png' % (round, i)), format='png')
            #
            # bw, mw = generate_delta_weights(i, benign_list_of_weights, malicious_list_of_weights, True)
            # create_plot(bw, mw, round, i)
            # plt.savefig(os.path.join(figures_dir, 'summed updates - round %s_layer %s.png' % (round, i)), format='png')

            # corr matrix
            bw, mw = generate_correlations(i, benign_list_of_weights, malicious_list_of_weights, True)
            print('bw vs mw corellation: ', (np.abs(bw)+1).prod(), (np.abs(mw)+1).prod())
            create_plot(bw, mw, round, i, cmap='hot')
            file_name = os.path.join(figures_dir, 'correlation_summed_updates_round_%s_layer_%i' % (round, i))
            plt.savefig(file_name+'.png', format='png',bbox_inches='tight')
            plt.savefig(file_name+'.pdf', format='pdf',bbox_inches='tight')

            # bw, mw = generate_correlations(i, benign_list_of_weights, malicious_list_of_weights, False)
            # create_plot(bw, mw, round, i, cmap='hot')
            # plt.savefig(os.path.join(figures_dir, 'correlation - rnd updates - round %s_layer %s.png' % (round, i)),
            #             format='png')

            break # since only visualization of the first layer is supported at this point


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--experiment_name", type=str, default='targeted_mal_baseline_boosting_true_num_malicious_clients_1_mnist_clients_10', help="Log file path.")
    parser.add_argument("--model_name", type=str, default='mnist_cnn', help="Which model to use.",
                        choices=['dev', 'mnist_cnn'])
    ARGS = parser.parse_args()

    main()
