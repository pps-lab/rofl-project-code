from glob import glob
from itertools import product

import pandas as pd
from os.path import join
from scipy.special import comb

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# matplotlib.use('Agg')


def get_colors(num_colors):
    colors = []
    for i in np.linspace(0, 1, num_colors):
        colors.append(plt.cm.RdYlBu(i))
    return list(reversed(colors))


def get_results(experiment_dir):
    log_file = join(experiment_dir, 'log.csv')
    df = pd.read_csv(log_file)

    return df['round'].values.tolist(), df['accuracy'].values.tolist(), df['adv_success'].values.tolist(),


def malicious_probability(K, M, C):
    return 1 - comb(K - M, C, exact=True) / comb(K, C, exact=True)


def main(num_clients, num_malicious):
    num_selected_clients = [10, 12, 14, 16, 18, 20]
    probability = [malicious_probability(num_clients, num_malicious, C) for C in range(10, 21)]
    experiment_files = glob(join(experiment_dir, 'num_selected_clients_*'))
    experiment_files = [file_name for file_name in experiment_files
                        if file_name.endswith('%i' % num_clients) and
                        ('num_malicious_clients_%i' % num_malicious) in file_name]

    plt.figure()
    for i, num_selected in enumerate(num_selected_clients):
        experiment_file = [x for x in experiment_files if ('num_selected_clients_%i' % num_selected) in x][0]
        rounds, _, adv_success = get_results(experiment_file)

        plt.plot(rounds, adv_success, label='C=%i, p=%.2f' % (num_selected, probability[i]), c=COLORS[i], linestyle='solid')

    plt.ylim(0., 1.)
    plt.xlabel('Rounds')
    plt.ylabel('Adversarial success')

    plt.legend()
    file_path = join('.', 'synergy_plots', f'num_clients_{num_clients}_num_malicious_{num_malicious}.pdf')
    plt.savefig(file_path)
    plt.show()


if __name__ == '__main__':
    COLORS = get_colors(6)
    experiment_dir = join('..', 'data_fetching', 'experiments')
    for C, M in product([25, 50], [1, 2]):
        main(C, M)
