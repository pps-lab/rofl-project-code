import json
from glob import glob
from itertools import product
import argparse

import pandas as pd
import os
import numpy as np
from os.path import join
import matplotlib

from matplotlib import pyplot as plt


def parse_log(log_file):
    df = pd.read_csv(log_file)
    accuracy = df['accuracy'].values.tolist()
    rounds = df['round'].values.tolist()
    adv_success = df['adv_success'].values.tolist()

    return accuracy, rounds, adv_success


def parse_config(config_file):
    with open(config_file) as f:
        json_data = f.read()
        config_dict = (json.loads(json_data))
        return config_dict


def get_df(directory_wildcard='hist*'):
    log_dir = join('experiments')
    dirs = glob(join(log_dir, ARGS.experiment_name))

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

    for exp_dir in dirs:
        accuracy, rounds, adv_success = parse_log(glob(join(exp_dir, 'log.csv'))[0])
        config_dict = parse_config(glob(join(exp_dir, 'config.json'))[0])

        table_values['accuracy'] += accuracy
        table_values['adv_success'] += adv_success
        table_values['round'] += rounds
        for key in ['dataset', 'num_clients', 'number_of_samples', 'attack_type', 'scale_attack', 'scale_attack_weight', 'num_malicious_clients']:
            table_values[key] = table_values[key] + [config_dict[key]] * len(accuracy)

    df = pd.DataFrame(table_values)
    return df


def make_plots(df, dataset, number_of_samples, attack_type=None, file_prefix='baseline',
               plot_dir='plotting/baseline_plots', ylabel='Accuracy', objective='accuracy'):
    matplotlib.use('Agg')
    colors = ['r', 'g', 'b', 'black']

    plt.figure()
    for i, num_clients in enumerate(ARGS.clients):
        data = df[(df.dataset == dataset) &
                  (df.number_of_samples == number_of_samples) &
                  (df.num_clients == num_clients)]
        if attack_type is not None:
            data = data[data.attack_type == attack_type]

        data = data.sort_values(by=['round'])
        accuracy, rounds = data[objective].values, data['round'].values

        plt.plot(rounds, accuracy, label='K=%i' % num_clients, c=colors[i])

    plt.ylim(0., 1.)
    plt.xlabel('Rounds')
    plt.ylabel(ylabel)
    plt.legend()

    if number_of_samples == -1:
        number_of_samples = 60000
    file_path = join('.', plot_dir, f'{file_prefix}_{dataset}_{number_of_samples}.pdf')
    plt.savefig(file_path)
    # plt.show()


def print_table(df, dataset, number_of_samples, objective='accuracy'):

    data = df.copy()
    data = data[(data.dataset == dataset) & (data.number_of_samples == number_of_samples)]

    to_print = ['Round & K=10 & K=25 & K=50 & K=100']
    for round in range(1, 41):
        round_data = data[data['round'] == round]
        to_app = ['%i.' % round]
        for num_clients in ARGS.clients:
            accuracy = round_data[round_data.num_clients == num_clients][objective].values[0]
            to_app.append('%.2f' % accuracy)
        to_print.append('&'.join(to_app))

    sep = r'\\\hline'
    to_print = sep.join(to_print) + sep
    print(to_print)


def main():

    df = get_df()

    for dataset, number_of_samples in product([ARGS.dataset], ARGS.samples):
        print(number_of_samples)
        print_table(df, dataset, number_of_samples)
        make_plots(df, dataset, number_of_samples)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--experiment_name", type=str,
                        default='targeted_mal_baseline_boosting_true_num_malicious_clients_1_mnist_clients_25',
                        help="Log file path.")
    parser.add_argument("--dataset", type=str, default='mnist', help="Which dataset to use.",
                        choices=['mnist', 'fmnist'])
    parser.add_argument("--samples", type=int, nargs="+", help="What number of samples to evaluate.")
    parser.add_argument("--clients", type=int, nargs="+", help="Number of clients.")

    ARGS = parser.parse_args()

    main()
