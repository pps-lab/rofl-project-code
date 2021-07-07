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

from common import _preprocess, setup_plt, query_data, get_colorful_styles, output_dir, COLOR_GRAY, yaxis_formatter, get_markers
from operator import itemgetter

def load_data_femnist():

    query = {
        "meta.description": "FEMNIST increase attackers, pgd and blackbox"
    }

    docs_list = query_data(query)

    metrics = [pd.DataFrame.from_dict(doc) for doc in docs_list]
    for i, metric in enumerate(metrics):
        df = pd.DataFrame.from_dict({
            'round': metric['metrics']['round'],
            'accuracy': metric['metrics']['accuracy'],
            'adv_success': metric['metrics']['adv_success']
        })

        num_attackers = int(metric['hyperparameters']['args']['environment']['num_selected_malicious_clients']) \
            if 'num_selected_malicious_clients' in metric['hyperparameters']['args']['environment'] else \
            int(metric['hyperparameters']['args']['environment']['num_malicious_clients'])
        attack_method = "blackbox" if 'objective' in metric['hyperparameters']['args']['client']['malicious'] else "pgd"

        metrics[i] = _preprocess(df, f"{num_attackers}_{attack_method}")

    df = reduce(lambda left, right: pd.merge(left, right, on=['round'], how='outer'), metrics)
    df = df[df["round"] <= 500]

    return df

def load_data_cifar():

    query = {
        # "meta.description": "CIFAR increase attackers multiple runs"
        "meta.description": "CIFAR10 increase attackers and blackbox, but temp"
    }

    docs_list = query_data(query)

    metrics = [pd.DataFrame.from_dict(doc) for doc in docs_list]
    metrics_out = []
    found_one_15 = False
    for i, metric in enumerate(metrics):
        df = pd.DataFrame.from_dict({
            'round': metric['metrics']['round'],
            'accuracy': metric['metrics']['accuracy'],
            'adv_success': metric['metrics']['adv_success']
        })

        num_attackers = int(metric['hyperparameters']['args']['environment']['num_selected_malicious_clients']) \
            if 'num_selected_malicious_clients' in metric['hyperparameters']['args']['environment'] else \
            int(metric['hyperparameters']['args']['environment']['num_malicious_clients'])
        attack_method = "blackbox" if 'objective' in metric['hyperparameters']['args']['client']['malicious'] else "pgd"


        if attack_method == "pgd" and num_attackers == 15:
            if not found_one_15:
                metrics_out.append(_preprocess(df, f"{num_attackers}_{attack_method}"))
                found_one_15 = True
                # print(i, metric['hyperparameters']['args'])
        else:
            metrics_out.append(_preprocess(df, f"{num_attackers}_{attack_method}"))

    df = reduce(lambda left, right: pd.merge(left, right, on=['round'], how='outer'), metrics_out)
    df = df[df["round"] <= 520]

    return df

def build_plot(name, df, model, leftmost):

    if model == 'FEMNIST':
        configs = {
            '1': {'label': '1 (3.3%)', 'marker': get_markers()[0]},
            '2': {'label': '2 (6.6%)', 'marker': get_markers()[1]},
            '5': {'label': '5 (16.7%)', 'marker': get_markers()[2]},
            '8': {'label': '8 (27%)', 'marker': get_markers()[3]}
        }
        df = df[df["round"] <= 500]
    elif model == 'CIFAR10':
        # configs = {
        #     num: {'label': f"{num} ({num / 0.4})%", 'marker': None} for num in [1, 2, 5, 10, 15, 20]
        # }
        configs = {
            '1': {'label': '1 (2.5%)', 'marker': get_markers()[0]},
            '2': {'label': '2 (5%)', 'marker': get_markers()[1]},
            # '10': {'label': '10 (25%)', 'marker': 'o'},
            # '5': {'label': '5 (12.5%)', 'marker': 'o'},
            '20': {'label': '20 (50%)', 'marker': get_markers()[2]}
        }


    markevery = 100
    window_size = 20

    setup_plt(square=True)

    with PdfPages(f"{output_dir}/{name}.pdf") as pdf:

        fig, ax = plt.subplots()
        colors, linestyles = get_colorful_styles()

        custom_lines_colors = []
        custom_lines_colors_names = []
        for index, suffix in enumerate(configs.keys()):
            values_pgd = df[f"adv_success_{suffix}_pgd"]
            values_blackbox = df[f"adv_success_{suffix}_blackbox"]
            labels = df[f"round"]
            config = configs[suffix]

            plt.plot(labels, values_pgd.rolling(window_size).mean().shift(-window_size),
                     linestyle='solid', label=config['label'], color=colors[index],
                     linewidth=2, marker=config['marker'], markevery=markevery)
            plt.plot(labels, values_blackbox.rolling(window_size).mean().shift(-window_size),
                     linestyle='dotted',
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
        ax.set_ylim(ymin=-0.05, ymax=1.05)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.get_yaxis().set_major_formatter(
            matplotlib.ticker.FuncFormatter(yaxis_formatter))

        if leftmost:
            plt.ylabel('Accuracy')
        plt.xlabel('Round')

        line = Line2D([0], [0])
        line.set_visible(False)
        custom_lines_colors = [line] + custom_lines_colors
        custom_lines_colors_names = ['# Att. ($\\alpha$):'] + custom_lines_colors_names
        if model == 'FEMNIST':
            ordering = [0, 3, 1, 4, 2]

            leg1 = plt.legend(itemgetter(*ordering)(custom_lines_colors),
                              itemgetter(*ordering)(custom_lines_colors_names),
                              bbox_to_anchor=(1.02, 1.), loc=4, ncol=3, columnspacing=0.75)
        elif model == 'CIFAR10':
            ordering = [0, 2, 1, 3]
            leg1 = plt.legend(itemgetter(*ordering)(custom_lines_colors),
                              itemgetter(*ordering)(custom_lines_colors_names),
                              bbox_to_anchor=(1.02, 1.), loc=4, ncol=2, columnspacing=0.75)

        ax.add_artist(leg1)

        if not leftmost:
            custom_lines_styles = [Line2D([0], [0], linestyle="-", lw=2, color=COLOR_GRAY),
                                   Line2D([0], [0], linestyle=":", lw=2, color=COLOR_GRAY)]
            leg_task = plt.legend(custom_lines_styles, ["PGD", "Blackbox"],
                                  bbox_to_anchor=(1., 0.6), loc=4, ncol=1,
                                  columnspacing=0.75)
            ax.add_artist(leg_task)

        # plt.legend(title='Bound', mode="expand", loc="lower left", labelspacing=.05, bbox_to_anchor=(1.01, 0, .6, 0))

        pdf.savefig(bbox_inches='tight', pad_inches=0)
        plt.savefig(f"{output_dir}/{name}.png", bbox_inches='tight', pad_inches=0)
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
        leg2 = plt.legend(custom_lines_styles, ["Backdoor task", "Main task"],
                          bbox_to_anchor=(1., -0.04, 1., .102), loc=3, ncol=1,
                          columnspacing=0.75)
        ax.add_artist(leg1)
        ax.add_artist(leg2)

        # plt.legend(title='Bound', mode="expand", loc="lower left", labelspacing=.05, bbox_to_anchor=(1.01, 0, .6, 0))

        pdf.savefig(bbox_inches='tight', pad_inches=0)
        plt.savefig(f"{output_dir}/{name}.png", bbox_inches='tight', pad_inches=0)
        plt.close()

    return fig, df
