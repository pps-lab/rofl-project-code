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

from common import _preprocess, setup_plt, query_data, get_colorful_styles, output_dir, COLOR_GRAY, COLOR_NO_BOUND, yaxis_formatter, get_markers, get_colors_attackers
from common import LINESTYLE_AT, LINESTYLE_PGD, LINESTYLE_NT, LINESTYLE_DP, DP_PATHEFFECT
from common import LABEL_AT, LABEL_PGD, LABEL_NT, LABEL_DP

from operator import itemgetter

import matplotlib.patheffects as pe

def load_data_femnist(bound_type="l2"):

    if bound_type == "l2":
        query = {
            "$or": [
                {"meta.description": "FEMNIST increase attackers, pgd and blackbox"},
                {"meta.description": "FEMNIST increase attackers, anticipate and neurotoxin"},
                # {"meta.description": "FEMNIST increase attackers median"},
                # {"meta.description": "FEMNIST increase attackers median anticipate"}
                {"meta.description": "FEMNIST increase attackers l2 anticipate"}

            ]
        }
    else:
        query = {
            "$or": [
                # {"meta.description": "FEMNIST increase attackers, pgd and blackbox"},
                # {"meta.description": "FEMNIST increase attackers, anticipate and neurotoxin"},
                {"meta.description": "FEMNIST increase attackers median"},
                {"meta.description": "FEMNIST increase attackers median anticipate"}
                # {"meta.description": "FEMNIST increase attackers l2 anticipate"}
            ]
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
        attack_method = get_attack_method(metric)

        metrics[i] = _preprocess(df, f"{num_attackers}_{attack_method}")

    df = reduce(lambda left, right: pd.merge(left, right, on=['round'], how='outer'), metrics)
    df = df[df["round"] <= 520]

    return df

def load_data_cifar(bound_type):

    if bound_type == "median":
        query = {
            # "meta.description": "CIFAR increase attackers multiple runs"
            # "meta.description": "CIFAR10 increase attackers and blackbox, but temp"
            "meta.description": "CIFAR10 increase attackers median 1.5"
        }
    else:
        query = {
            # "meta.description": "CIFAR increase attackers multiple runs"
            "meta.description": "CIFAR10 increase attackers and blackbox, but temp"
            # "meta.description": "CIFAR10 increase attackers median 1.5"
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
        attack_method = get_attack_method(metric)


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

def get_attack_method(metric):
    if metric['client']['malicious']['evasion'] == None:
        return "blackbox"
    elif metric['client']['malicious']['objective']['name'] == 'TargetedAttack':
        if metric['client']['malicious']['evasion']['name'] == 'NormBoundPGDEvasion':
            return "pgd"
        elif metric['client']['malicious']['evasion']['name'] == 'NeurotoxinEvasion':
            return "neurotoxin"
        else:
            return "ERROR"
    elif metric['client']['malicious']['objective']['name'] == 'AnticipateTfAttack':
        return "anticipate"
    return "ERROR2"

def build_plot(name, df, model, leftmost, configs, attacks, ls_map):

    markevery = 100
    window_size = 20

    setup_plt(square=True)

    label_map = {
        'pgd': LABEL_PGD,
        'anticipate': LABEL_AT,
        'neurotoxin': LABEL_NT,
        'blackbox': LABEL_DP
    }

    with PdfPages(f"{output_dir}/{name}.pdf") as pdf:

        fig, ax = plt.subplots()

        custom_lines_colors = []
        custom_lines_colors_names = []
        for index, suffix in enumerate(configs.keys()):

            labels = df[f"round"]
            config = configs[suffix]

            for attack in attacks:
                values = df[f"adv_success_{suffix}_{attack}"]
                plt.plot(labels, values.rolling(window_size).mean().shift(-window_size),
                         linestyle=ls_map[attack], label=config['label'], color=config['color'],
                         linewidth=2, marker=config['marker'], markevery=markevery)


            custom_lines_colors.append(Line2D([0], [0], linestyle="-", lw=2, marker=config['marker'], color=config['color']))
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
            plt.ylabel('Accuracy ($\\%$)')
        plt.xlabel('Round')

        line = Line2D([0], [0])
        line.set_visible(False)
        custom_lines_colors = [line] + custom_lines_colors
        custom_lines_colors_names = ['# Att. ($\\alpha$):'] + custom_lines_colors_names
        if model == 'FEMNIST':
            ordering = [0, 3, 1, 4, 2, 5]
            custom_lines_colors.insert(3, line)
            custom_lines_colors_names.insert(3, '')

            leg1 = plt.legend(itemgetter(*ordering)(custom_lines_colors),
                              itemgetter(*ordering)(custom_lines_colors_names),
                              bbox_to_anchor=(1.02, 1.), loc=4, ncol=3, columnspacing=0.75)

            for vpack in leg1._legend_handle_box.get_children()[:1]:
                for hpack in vpack.get_children():
                    del hpack._children[0]
            ax.add_artist(leg1)
        elif model == 'CIFAR10':
            ordering = [0, 3, 1, 4, 2, 5]
            custom_lines_colors.insert(3, line)
            custom_lines_colors_names.insert(3, '')

            leg1 = plt.legend(itemgetter(*ordering)(custom_lines_colors),
                              itemgetter(*ordering)(custom_lines_colors_names),
                              bbox_to_anchor=(1.02, 1.), loc=4, ncol=3, columnspacing=0.75)
            for vpack in leg1._legend_handle_box.get_children()[:1]:
                for hpack in vpack.get_children():
                    del hpack._children[0]
            ax.add_artist(leg1)


        if leftmost:
            custom_lines_styles = [Line2D([0], [0], linestyle=ls_map[attack], lw=2, color=COLOR_GRAY) for attack in attacks]
            custom_labels = [label_map[attack] for attack in attacks]

            leg_task = plt.legend(custom_lines_styles, custom_labels,
                                  bbox_to_anchor=(0.025, 1.0), loc="upper left", ncol=2,
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
        ax.set_ylabel("Accuracy ($\\%$)")
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])

        plt.ylabel('Accuracy ($\\%$)')
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
