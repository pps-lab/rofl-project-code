from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import yaml
from matplotlib import cm
import pymongo
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

from common import _preprocess, setup_plt, query_data, get_colorful_styles, output_dir, COLOR_GRAY

def load_data_femnist():

    query = {
        "meta.description": "FEMNIST outlier handwriters (incomplete)"
    }

    docs_list = query_data(query)

    metrics = [pd.DataFrame.from_dict(doc) for doc in docs_list]
    metrics_df = []
    for i, metric in enumerate(metrics):
        if 'result' in metric['metrics']:
            continue

        df = pd.DataFrame.from_dict({
            'round': metric['metrics']['round'],
            'accuracy': metric['metrics']['accuracy'],
            'adv_success': metric['metrics']['adv_success']
        })

        attack_type = 'blackbox' if 'malicious' in metric['hyperparameters']['args']['client'] \
                            else 'pgd'  # pgd is default

        mal_client_indic = metric['hyperparameters']['args']['environment']['malicious_client_indices'][0]
        mal_client_map = {
            1716: 'outlier',
            2165: 'random',
            2589: 'prototype'
        }

        objective = mal_client_map[mal_client_indic]
        bound = metric['hyperparameters']['args']['client']['clip']['value'] \
            if metric['hyperparameters']['args']['client']['clip'] is not None else None
        metrics_df.append(_preprocess(df, f"{attack_type}_{objective}_{bound}"))

    df = reduce(lambda left, right: pd.merge(left, right, on=['round'], how='outer'), metrics_df)
    df = df[df["round"] <= 400]

    return df

def build_plot(name, df, model):

    if model == 'FEMNIST':
        configs = {
            'None_blackbox_noattack': {'label': 'None', 'marker': None},
            'edge_pgd': {'label': 'Tail (PGD)', 'marker': 'v'},
            'edge_blackbox': {'label': 'Tail ARDIS (Blackbox)', 'marker': 'v'},
            'tasks_pgd': {'label': 'Poison handwriters (PGD)', 'marker': 's'},
            'tasks_blackbox': {'label': 'Poison handwriters (Blackbox)', 'marker': 's'},
        }
        bounds = [1.0, 2.0, 4.0, 8.0]
        trainings = ['pgd',
                     'blackbox']
        attack_types = ['outlier', 'random', 'prototype']
    elif model == 'CIFAR10':
        configs = {
            'bgwall': {'label': 'A1-WALL', 'marker': 's'},
            'greencar': {'label': 'A2-GREEN', 'marker': 'o'},
            'racingstripe': {'label': 'A3-STRIPE', 'marker': 'v'}
        }

    markers = ['s', 'o', 'v']
    linestyles = {'pgd': 'solid', 'blackbox': 'dotted'}
    attack_labels = {'prototype': 'Prototypes', 'outlier': 'Outliers', 'random': 'Random'}


    markevery = 100
    window_size = 21

    setup_plt()

    with PdfPages(f"{output_dir}/{name}.pdf") as pdf:

        fig, ax = plt.subplots()
        colors, _ = get_colorful_styles()

        custom_lines_colors = [Line2D([0], [0], linestyle="-", lw=2, marker=markers[index], color=colors[index])
                               for index in range(3)]
        custom_lines_colors_names = [attack_labels[type] for type in attack_types]

        for train in trainings:
            for index, attack_type in enumerate(attack_types):
                labels = bounds
                mean_adv = [df[f"adv_success_{train}_{attack_type}_{bound}"][
                                df[f"adv_success_{train}_{attack_type}_{bound}"]
                            .notna()].tail(20).mean() for bound in bounds]

                plt.plot(labels, mean_adv, color=colors[index], linestyle=linestyles[train], linewidth=2, marker=markers[index])

        # for index, suffix in enumerate(configs.keys()):
        #     values_acc = df[f"accuracy_{suffix}"]
        #     values_adv = df[f"adv_success_{suffix}"]
        #     labels = df[f"round"]
        #     config = configs[suffix]
        #
        #     plt.plot(labels, values_acc.rolling(window_size).mean(),
        #              linestyle='dotted', label=config['label'], color=colors[index],
        #              linewidth=2, marker=config['marker'], markevery=markevery)
        #     plt.plot(labels, values_adv.rolling(window_size).mean(),
        #              color=colors[index],
        #              linewidth=2, marker=config['marker'], markevery=markevery)
        #     custom_lines_colors.append(Line2D([0], [0], linestyle="-", lw=2, marker=config['marker'], color=colors[index]))
        #     custom_lines_colors_names.append(config['label'])

        ##########################
        # General Format
        ##########################
        ax.grid(True, axis="y", linestyle=':', color='0.6', zorder=0, linewidth=1.2)
        ##########################
        # Y - Axis Format
        ##########################
        ax.set_ylim(ymin=0, ymax=1.01)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])

        plt.ylabel('Adversarial success')
        plt.xlabel('Norm bound ($L_2$)')

        # Legend
        custom_lines_styles = [Line2D([0], [0], linestyle=linestyles['pgd'], lw=2, color=COLOR_GRAY),
                               Line2D([0], [0], linestyle=linestyles['blackbox'], lw=2, color=COLOR_GRAY)]
        leg1 = plt.legend(custom_lines_colors, custom_lines_colors_names,
                          bbox_to_anchor=(1., 0.28, 1., .102), loc=3, ncol=1, columnspacing=0.75, title="Backdoor input")
        leg2 = plt.legend(custom_lines_styles, ["PGD", "Blackbox"],
                          bbox_to_anchor=(1., -0.04, 1., .102), loc=3, ncol=1,
                          columnspacing=0.75)
        ax.add_artist(leg1)
        ax.add_artist(leg2)

        # plt.legend(title='Bound', mode="expand", loc="lower left", labelspacing=.05, bbox_to_anchor=(1.01, 0, .6, 0))

        pdf.savefig(bbox_inches='tight', pad_inches=0)
        plt.close()

    return fig, df


def build_continuous_median_plot(name, df, model):

    if model == 'FEMNIST':
        configs = {
            'None_None_None': {'label': 'None', 'marker': None},
            'median_l2_1.0_1.0': {'label': '1.0', 'marker': 's'},
            'median_l2_5.0_5.0': {'label': '5.0', 'marker': 'o'},
            'median_l2_10.0_10.0': {'label': '10.0', 'marker': 'v'}
        }
    elif model == 'CIFAR10':
        configs = {
            'None_None_None': {'label': 'None', 'marker': None},
            'l2_5.0_0.07352941176470588': {'label': '5.0', 'marker': 's'},
            'l2_10.0_0.14705882352941177': {'label': '10.0', 'marker': 'o'},
            'l2_15.0_0.19607843137254902': {'label': '15.0', 'marker': 'v'}
        }


    markevery = 100
    window_size = 20

    setup_plt()

    with PdfPages(f"{output_dir}/{name}.pdf") as pdf:

        fig, ax = plt.subplots()
        colors, linestyles = get_colorful_styles()

        custom_lines_colors = []
        custom_lines_colors_names = []
        for index, suffix in enumerate(configs.keys()):
            values_acc = df[f"accuracy_{suffix}"]
            values_adv = df[f"adv_success_{suffix}"]
            labels = df[f"round"]
            config = configs[suffix]

            plt.plot(labels, values_acc.rolling(window_size).mean(),
                     linestyle='dotted', label=config['label'], color=colors[index],
                     linewidth=2, marker=config['marker'], markevery=markevery)
            plt.plot(labels, values_adv.rolling(window_size).mean(),
                     color=colors[index],
                     linewidth=2, marker=config['marker'], markevery=markevery)
            custom_lines_colors.append(Line2D([0], [0], linestyle="-", lw=2, marker=config['marker'], color=colors[index]))
            custom_lines_colors_names.append(config['label'])

        ##########################
        # General Format
        ##########################
        ax.grid(True, axis="y", linestyle=':', color='0.6', zorder=0, linewidth=1.2)
        ##########################
        # Y - Axis Format
        ##########################
        ax.set_ylim(ymin=0, ymax=1.01)
        ax.set_ylabel("Accuracy")
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])

        plt.ylabel('Accuracy')
        plt.xlabel('Round')

        # Legend
        custom_lines_styles = [Line2D([0], [0], linestyle="-", lw=2, color=COLOR_GRAY),
                               Line2D([0], [0], linestyle=":", lw=2, color=COLOR_GRAY)]
        leg1 = plt.legend(custom_lines_colors, custom_lines_colors_names,
                          bbox_to_anchor=(1., 0.28, 1., .102), loc=3, ncol=1, columnspacing=0.75, title="Median bound")
        leg2 = plt.legend(custom_lines_styles, ["Main task", "Backdoor task"],
                          bbox_to_anchor=(1., -0.04, 1., .102), loc=3, ncol=1,
                          columnspacing=0.75)
        ax.add_artist(leg1)
        ax.add_artist(leg2)

        # plt.legend(title='Bound', mode="expand", loc="lower left", labelspacing=.05, bbox_to_anchor=(1.01, 0, .6, 0))

        pdf.savefig(bbox_inches='tight', pad_inches=0)
        plt.close()

    return fig, df
