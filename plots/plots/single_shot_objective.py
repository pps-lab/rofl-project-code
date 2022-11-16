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

from common import _preprocess, setup_plt, query_data, get_colorful_styles, output_dir, COLOR_GRAY, COLOR_NO_BOUND, yaxis_formatter, get_markers, get_colors_attack_objective
from common import LINESTYLE_AT, LINESTYLE_PGD, LINESTYLE_NT, LINESTYLE_DP
from common import LABEL_AT, LABEL_PGD, LABEL_NT, LABEL_DP

def load_data_femnist():

    query = {
        # "meta.description": "FEMNIST single-shot under bound (and edge case)",
        # "hyperparameters.args.client.malicious.backdoor.type": {'$ne': 'edge'}
        "$or": [
            {"meta.description": "FEMNIST single shot new attacks"},
            {"meta.description": "FEMNIST single shot anticipate"},
            {"meta.description": "FEMNIST single shot median 1.5"},
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
        not_attacking = 'attack_stop' in metric['hyperparameters']['args']['client']['malicious'] \
            if 'malicious' in metric['hyperparameters']['args']['client'] is not None else None
        attack_method  = get_attack_method(metric)
        backdoor_type = get_backdoor_type(metric)
        clip_type = metric['hyperparameters']['args']['client']['clip']['type'] \
                            if metric['hyperparameters']['args']['client']['clip'] is not None else None
        bound = metric['hyperparameters']['args']['client']['clip']['value'] \
                            if metric['hyperparameters']['args']['client']['clip'] is not None else None
        metrics[i] = _preprocess(df, f"{backdoor_type}_{clip_type}_{bound}_{attack_method}")

    df = reduce(lambda left, right: pd.merge(left, right, on=['round'], how='outer'), metrics)
    df = df[df["round"] <= 150]
    return df

def load_data_cifar():
    query = {
        # "meta.description": "CIFAR10 single shot comparison of edge cases and defenses"
        "$or": [
            {"meta.description": "CIFAR10 single shot comparison of edge cases and defenses"},
            {"meta.description": "CIFAR10 single shot new attacks"},
            {"meta.description": "CIFAR10 single shot median 1.5"}
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
        attack_method = get_attack_method(metric)
        backdoor_type = get_backdoor_type(metric)
        clip_type = metric['hyperparameters']['args']['client']['clip']['type'] \
                            if 'clip' in metric['hyperparameters']['args']['client'] and \
                               metric['hyperparameters']['args']['client']['clip'] is not None else None
        bound = metric['hyperparameters']['args']['client']['clip']['value'] \
                            if 'clip' in metric['hyperparameters']['args']['client'] is not None else None
        metrics[i] = _preprocess(df, f"{backdoor_type}_{clip_type}_{bound}_{attack_method}")

    df = reduce(lambda left, right: pd.merge(left, right, on=['round'], how='outer'), metrics)
    df = df[df["round"] <= 25]
    return df


def get_backdoor_type(metric):
    if 'malicious' not in metric['hyperparameters']['args']['client']:
        return None
    if 'backdoor' not in metric['hyperparameters']['args']['client']['malicious']:
        return None
    if 'type' not in metric['hyperparameters']['args']['client']['malicious']['backdoor']:
        return 'semantic'
    elif metric['hyperparameters']['args']['client']['malicious']['backdoor']['type'] == 'edge':
        return metric['hyperparameters']['args']['client']['malicious']['backdoor']['edge_case_type']
    else:
        return metric['hyperparameters']['args']['client']['malicious']['backdoor']['type']

def get_attack_method(metric):
    # if metric['client']['malicious']['attack']
    if metric['client']['malicious']['evasion'] == None:
        return 'blackbox'
    elif metric['client']['malicious']['objective']['name'] == "TargetedAttack":
        if metric['client']['malicious']['evasion']['name'] == "NeurotoxinEvasion":
            return 'neurotoxin'
        else:
            return 'pgd'
    elif metric['client']['malicious']['objective']['name'] == "AnticipateTfAttack":
        return 'anticipate'
    else:
        return 'ERROR'

def build_plot(name, df, model, leftmost):
    print("BUIDLPLOT")

    colors = get_colors_attack_objective()
    markers = get_markers()

    if model == 'FEMNIST':
        # configs = {
        #     ('FEMNISTRandomNoiseEdgeCase_None_None_pgd', 'FEMNISTRandomNoiseEdgeCase_l2_4.0'): {'label': 'Noise', 'marker': markers[0], 'color': colors[0]},
        #     ('EuropeanSevenEdgeCase_None_None_pgd', 'EuropeanSevenEdgeCase_l2_4.0'): {'label': 'Tail', 'marker': markers[1], 'color': colors[1]},
        #     ('None_None_None_pgd', 'None_l2_4.0'): {'label': 'Prototype', 'marker': markers[2], 'color': colors[2]}
        # }
        configs = {
            ('FEMNISTRandomNoiseEdgeCase_None_None_pgd', 'FEMNISTRandomNoiseEdgeCase_median_l2_1.5'): {'label': 'Noise', 'marker': markers[0], 'color': colors[0]},
            ('EuropeanSevenEdgeCase_None_None_pgd', 'EuropeanSevenEdgeCase_median_l2_1.5'): {'label': 'Tail', 'marker': markers[1], 'color': colors[1]},
            ('None_None_None_pgd', 'None_median_l2_1.5'): {'label': 'Prototype', 'marker': markers[2], 'color': colors[2]}
        }
        attacks = ['pgd', 'anticipate', 'neurotoxin']
    elif model == 'CIFAR10':
        # configs = {
        #     ('CifarRandomNoiseEdgeCase_None_None_pgd', 'CifarRandomNoiseEdgeCase_l2_5.0'): {'label': 'Noise', 'marker': markers[0], 'color': colors[0]},
        #     ('NorthWesternEdgeCase_None_None_pgd', 'NorthWesternEdgeCase_l2_5.0'): {'label': 'Tail', 'marker': markers[1], 'color': colors[1]},
        #     ('semantic_None_None_pgd', 'semantic_l2_5.0'): {'label': 'Prototype', 'marker': markers[2], 'color': colors[2]},
        # }
        configs = {
            ('CifarRandomNoiseEdgeCase_None_None_pgd', 'CifarRandomNoiseEdgeCase_median_l2_1.5'): {'label': 'Noise', 'marker': markers[0], 'color': colors[0]},
            ('NorthWesternEdgeCase_None_None_pgd', 'NorthWesternEdgeCase_median_l2_1.5'): {'label': 'Tail', 'marker': markers[1], 'color': colors[1]},
            ('semantic_None_None_pgd', 'semantic_median_l2_1.5'): {'label': 'Prototype', 'marker': markers[2], 'color': colors[2]},
        }
        attacks = ['pgd', 'anticipate', 'neurotoxin']


    markevery = 30 if model == 'FEMNIST' else 5
    window_size = 1

    setup_plt(square=True)

    linestyle_attack_map = {
        'pgd': LINESTYLE_PGD,
        'anticipate': LINESTYLE_AT,
        'neurotoxin': LINESTYLE_NT
    }

    with PdfPages(f"{output_dir}/{name}.pdf") as pdf:

        fig, ax = plt.subplots()

        custom_lines_colors = []
        custom_lines_colors_names = []
        for index, (suf_unprotected, suf_defense) in enumerate(configs.keys()):
            values_no_defense = df[f"adv_success_{suf_unprotected}"]
            labels = df[f"round"]
            config = configs[(suf_unprotected, suf_defense)]

            plt.plot(labels, values_no_defense.rolling(window_size).mean(),
                     color=config['color'],
                     linewidth=2, marker=config['marker'], markevery=markevery)

            for attack in attacks:
                values_defended = df[f"adv_success_{suf_defense}_{attack}"]
                plt.plot(labels, values_defended.rolling(window_size).mean(),
                         linestyle=linestyle_attack_map[attack], label=config['label'], color=config['color'],
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

        # Legend
        if leftmost:
            line = Line2D([0], [0])
            line.set_visible(False)
            custom_lines_colors = [line] + custom_lines_colors
            custom_lines_colors_names = ['Att. Obj.:'] + custom_lines_colors_names

            leg1 = plt.legend(custom_lines_colors, custom_lines_colors_names,
                              bbox_to_anchor=(-0.025, 1.16), loc='lower left', ncol=6, columnspacing=0.75)
            ax.add_artist(leg1)

            # if leftmost:
            for vpack in leg1._legend_handle_box.get_children()[:1]:
                for hpack in vpack.get_children():
                    del hpack._children[0]

        if leftmost:
            line = Line2D([0], [0])
            line.set_visible(False)
            custom_lines_styles = [
                line,
                Line2D([0], [0], linestyle=LINESTYLE_AT, lw=2, color=COLOR_GRAY),
                Line2D([0], [0], linestyle=LINESTYLE_PGD, lw=2, color=COLOR_GRAY),
                Line2D([0], [0], linestyle=LINESTYLE_NT, lw=2, color=COLOR_GRAY),
                Line2D([0], [0], linestyle=LINESTYLE_DP, lw=2, color=COLOR_GRAY)
            ]
            labels = [
                "       M-$r$ :",
                f"Bound ({LABEL_AT})",
                f"Bound ({LABEL_PGD})",
                f"Bound ({LABEL_NT})",
                "No Bound  ",
            ]

            # labels = ["No defense"]
            # labels = labels + [f"Norm bound {attack}" for attack in attacks]
            # for attack in attacks:
            #     labels.append(f"Norm bound {attack}")

            leg_task = plt.legend(custom_lines_styles, labels,
                                  bbox_to_anchor=(-0.025, 0.99), loc='lower left', ncol=6,
                                  columnspacing=0.75)
            for vpack in leg_task._legend_handle_box.get_children()[:1]:
                for hpack in vpack.get_children():
                    del hpack._children[0]
            ax.add_artist(leg_task)

        pdf.savefig(bbox_inches='tight', pad_inches=0)
        plt.savefig(f"{output_dir}/{name}.png", bbox_inches='tight', pad_inches=0)
        plt.close()

    return fig, df
