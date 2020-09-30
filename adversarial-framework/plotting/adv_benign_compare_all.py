import argparse
from glob import glob
from os.path import join

import matplotlib
from matplotlib import pyplot as plt
import pandas as pd

import tkinter as tk
from tkinter import filedialog

from plotting.baseline import parse_log, parse_config


def main():

    dfs = get_df_list("", ARGS.experiment_name)
    make_plots(dfs)


def make_plots(dfs):
    matplotlib.use('Agg')
    colors = ['r', 'g', 'b', 'black']

    plot_single = ARGS.plot_single == 'true'

    if plot_single:
        plt.figure()

    for obj in dfs:

        if not plot_single:
            plt.figure()

        data = obj["df"].sort_values(by=['round'])

        dataset = obj["config"]["dataset"]
        experiment_name = obj["config"]["experiment_name"]
        number_of_samples = obj["config"]["number_of_samples"]
        attack_type = obj["config"]["attack_type"]

        accuracy, rounds = data['accuracy'].values, data['round'].values
        plt.plot(rounds, accuracy, label=f'Model objective {experiment_name}')

        adv, rounds = data['adv_success'].values, data['round'].values
        plt.plot(rounds, adv, label=f'Adversarial Success {experiment_name}')

        if not plot_single:
            plt.ylim(0., 1.)
            plt.xlabel('Rounds')
            plt.ylabel('Accuracy')
            plt.legend()

            if number_of_samples == -1:
                number_of_samples = 60000
            file_path = join('.', 'plotting', 'combined_adv_acc_plots', f'{experiment_name}_{attack_type}_{dataset}_{number_of_samples}.png')
            plt.savefig(file_path)
            plt.show()

    if plot_single:
        plt.ylim(-0.1, 1.1)
        plt.xlabel('Rounds')
        plt.ylabel('Accuracy')
        plt.legend()

        file_path = join('.', 'plotting', 'combined_adv_acc_plots',
                         f'combined_all.png')
        plt.savefig(file_path)
        plt.show()


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

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--experiment_name", type=str,
                        default='targeted_mal_baseline_boosting_true_num_malicious_clients_1_mnist_clients_25',
                        help="Log file path. Accepts wildcards")
    parser.add_argument("--plot_single", type=str, default='false', help="Whether to plot all in single polot.",
                        )

    ARGS = parser.parse_args()

    main()
