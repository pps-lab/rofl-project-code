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
        "meta.description": "FEMNIST show comparison of pgd and blackbox for edge cases and non edge cases"
    }

    docs_list = query_data(query)

    metrics = [pd.DataFrame.from_dict(doc) for doc in docs_list]
    for i, metric in enumerate(metrics):
        df = pd.DataFrame.from_dict({
            'round': metric['metrics']['round'],
            'accuracy': metric['metrics']['accuracy'],
            'adv_success': metric['metrics']['adv_success']
        })
        not_attacking = 'attack_stop' in metric['hyperparameters']['args']['client']['malicious'] \
            if 'malicious' in metric['hyperparameters']['args']['client'] is not None else None
        backdoor_type = metric['hyperparameters']['args']['client']['malicious']['backdoor']['type'] \
                            if 'backdoor' in metric['hyperparameters']['args']['client']['malicious'] is not None else None
        attack_type = 'pgd' if 'objective' in metric['hyperparameters']['args']['client']['malicious'] \
                            else 'blackbox'  # blackbox is default
        edge_case_p = metric['hyperparameters']['args']['client']['malicious']['backdoor']['edge_case_p'] \
                            if 'backdoor' in metric['hyperparameters']['args']['client']['malicious'] and\
                               'edge_case_p' in metric['hyperparameters']['args']['client']['malicious']['backdoor'] is not None else None
        clip_type = metric['client']['clip']['type'] \
                            if 'clip' in metric['client'] and \
                               metric['client']['clip'] is not None else None
        metrics[i] = _preprocess(df, f"{backdoor_type}_{attack_type}_{edge_case_p}_{clip_type}{'_noattack' if not_attacking else ''}")

    df = reduce(lambda left, right: pd.merge(left, right, on=['round'], how='outer'), metrics)

    return df

def load_data_cifar():
    query = {
        "meta.description": "CIFAR single shot comparison of different objectives"
    }

    docs_list = query_data(query)

    metrics = [pd.DataFrame.from_dict(doc) for doc in docs_list]
    for i, metric in enumerate(metrics):
        df = pd.DataFrame.from_dict({
            'round': metric['metrics']['round'],
            'accuracy': metric['metrics']['accuracy'],
            'adv_success': metric['metrics']['adv_success']
        })
        backdoor_test = metric['hyperparameters']['args']['client']['malicious']['backdoor']['test'][0]
        backdoor_map = {
            330: "bgwall",
            42119: "racingstripe",
            41336: "greencar"
        }
        metrics[i] = _preprocess(df, backdoor_map[backdoor_test])

    df = reduce(lambda left, right: pd.merge(left, right, on=['round'], how='outer'), metrics)
    df = df[df["round"] <= 30]
    return df

def build_plot(name, df, model):

    if model == 'FEMNIST':
        configs = {
            'None_blackbox_noattack': {'label': 'None', 'marker': None, 'color_index': 0},
            'edge_pgd': {'label': 'Edge-case ARDIS (PGD)', 'marker': 'v', 'color_index': 1},
            'edge_blackbox': {'label': 'Edge-case ARDIS (Blackbox)', 'marker': 'v', 'color_index': 2},
            'tasks_pgd': {'label': 'Poison handwriters (PGD)', 'marker': 's', 'color_index': 1},
            'tasks_blackbox': {'label': 'Poison handwriters (Blackbox)', 'marker': 's', 'color_index': 2},
        }
    elif model == 'CIFAR10':
        configs = {
            'bgwall': {'label': 'A1-WALL', 'marker': 's'},
            'greencar': {'label': 'A2-GREEN', 'marker': 'o'},
            'racingstripe': {'label': 'A3-STRIPE', 'marker': 'v'}
        }


    markevery = 100
    window_size = 21

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
                     linestyle='dotted', label=config['label'], color=colors[config['color_index']],
                     linewidth=2, marker=config['marker'], markevery=markevery)
            plt.plot(labels, values_adv.rolling(window_size).mean(),
                     color=colors[config['color_index']],
                     linewidth=2, marker=config['marker'], markevery=markevery)
            custom_lines_colors.append(Line2D([0], [0], linestyle="-", lw=2, marker=config['marker'], color=colors[config['color_index']]))
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
                          bbox_to_anchor=(1., 0.28, 1., .102), loc=3, ncol=1, columnspacing=0.75, title="Attack target & training method")
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
        plt.close()

    return fig, df
