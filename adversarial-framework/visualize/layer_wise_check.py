import argparse
from glob import glob
from os.path import join
import matplotlib.pyplot as plt
import numpy as np

from src.tf_model import Model
from visualize.utils import get_model_name
from visualize.visualize_updates import get_directories, split_mal_ben_client_files


def sum_updates(files, normalize_updates=True):
    """Sums updates. Normalization is wrt number of parameters per layer."""
    updates = [np.load(file, allow_pickle=True) for file in files]
    to_ret = updates[0]
    for update in updates[1:]:
        to_ret += update

    if normalize_updates:
        for layer in range(to_ret.shape[0]):
            to_ret[layer] /= to_ret[layer].size
    return to_ret


def layer_norm(update, model_name):
    """Returns a list of norms for each layer (biases are discarded)."""
    model = Model.create_model(model_name)
    model.set_weights(update)
    norm_list = []
    for i in range(len(model.layers)):
        w = model.layers[i].get_weights()
        if w:
            w = w[0].flatten()  # 0 is weights, 1 is biases
            norm_list.append(np.linalg.norm(w))
    return norm_list


def plot_norms(b_norms, m_norms, g_norms, round_vec=None):
    """Generates a line chart for provided vectors of normals.

    Args:
        b_norms (np.ndarray): 2D array of l2 norm values for benign updates  (shape = (num of rounds, num of layers).
        m_norms (np.ndarray): 2D array of l2 norm values for malicious updates (shape = (num of rounds, num of layers).
        g_norms (np.ndarray): 2D array of l2 norm values for global model updates (shape = (num of rounds, num of layers).
        round_vec (np.ndarray or list): Rounds that are plotted on the x-axis.
    """
    linestyles = ['-', '--', '-.', ':']
    colors = ['r', 'g', 'b', 'purple']
    rounds, number_of_layers = b_norms.shape

    if round_vec is None:
        round_vec = range(rounds)

    def _plot_update_norms(norms):
        for layer in range(number_of_layers):
            plt.plot(round_vec, norms[:, layer],
                     # linestyle=linestyles[layer],
                     c=colors[layer], label=layer)

    plt.figure()

    plt.subplot(3, 1, 1)
    plt.title('Benign updates')
    _plot_update_norms(b_norms)
    plt.ylabel('L2 norm')

    plt.subplot(3, 1, 2)
    plt.title('Malicious updates')
    _plot_update_norms(m_norms)
    plt.ylabel('L2 norm')

    plt.subplot(3, 1, 3)
    plt.title('Global updates')
    _plot_update_norms(g_norms)
    plt.legend(title='Layers')
    plt.xlabel('Round')
    plt.ylabel('L2 norm')

    plt.tight_layout()


def main():
    updates_dir, figures_dir = get_directories(ARGS.experiment_name)
    model_name = get_model_name(ARGS.experiment_name)

    files = glob(join(updates_dir, '*.npy'))

    b_norms, m_norms, g_norms = [], [], []
    round_vec = list(range(ARGS.round_range[0], ARGS.round_range[1], 1))
    for round in round_vec:
        b_files, m_files = split_mal_ben_client_files(files, round)
        b_update, m_update = sum_updates(b_files), sum_updates(m_files)

        b_norms.append(layer_norm(b_update, model_name))
        m_norms.append(layer_norm(m_update, model_name))
        g_norms.append(layer_norm(b_update + m_update, model_name))

    b_norms, m_norms, g_norms = np.array(b_norms), np.array(m_norms), np.array(g_norms)
    plot_norms(b_norms, m_norms, g_norms, round_vec)
    plt.savefig(join(figures_dir, 'tmp.png'), format='png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default='visualization',
                        help="Sub-directory where the log files are stored.")
    parser.add_argument("--round_range", type=int, default=[10, 17], nargs='+',
                        help="Specify range for which rounds to generate layer-norm line chart.")
    ARGS = parser.parse_args()

    assert len(ARGS.round_range) == 2, "Round range must be a tuple of two integers."

    main()
