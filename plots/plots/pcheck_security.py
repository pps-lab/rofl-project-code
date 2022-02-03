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

def load_data_femnist():

    query = {
        "meta.description": "FEMNIST probabilistic checking topk (trial)"
    }

    docs_list = query_data(query)

    metrics = [pd.DataFrame.from_dict(doc) for doc in docs_list]
    transformed = []
    for i, metric in enumerate(metrics):
        df = pd.DataFrame.from_dict({
            'round': metric['metrics']['round'],
            'accuracy': metric['metrics']['accuracy'],
            'adv_success': metric['metrics']['adv_success']
        })
        interesting_round = df[df["round"] == 5.0]
        keep_number_of_weights = metric['hyperparameters']['args']['client']['malicious']['evasion']['args']['keep_number_of_weights']
        total_weights = 44426
        transformed.append({
            'accuracy': interesting_round["accuracy"].item(),
            'adv_success': interesting_round["adv_success"].item(),
            'keep_number_of_weights': keep_number_of_weights / total_weights
        })

    df = pd.DataFrame(transformed)
    return df

def load_data_cifar():
    query = {
        "meta.description": "CIFAR10 Probabilistic Checking security for topk attack"
    }

    docs_list = query_data(query)

    metrics = [pd.DataFrame.from_dict(doc) for doc in docs_list]
    transformed = []
    for i, metric in enumerate(metrics):
        df = pd.DataFrame.from_dict({
            'round': metric['metrics']['round'],
            'accuracy': metric['metrics']['accuracy'],
            'adv_success': metric['metrics']['adv_success']
        })
        interesting_round = df[df["round"] == 5.0]
        keep_number_of_weights = metric['hyperparameters']['args']['client']['malicious']['evasion']['args']['keep_number_of_weights']
        backdoor_type = 'tail' if "edge_case_type" in metric['client']['malicious']['backdoor'] else 'prototype'
        if backdoor_type != 'prototype':
            continue

        total_weights = 273066
        transformed.append({
            'accuracy': interesting_round["accuracy"].item(),
            'adv_success': interesting_round["adv_success"].item(),
            'keep_number_of_weights': keep_number_of_weights / total_weights,
            # 'backdoor_type': backdoor_type
        })

    df = pd.DataFrame(transformed)
    df = df.sort_values(by='keep_number_of_weights')
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


def build_plot(name, df, model, leftmost):

    if model == 'FEMNIST':
        configs = {
            ('EuropeanSevenEdgeCase_None_None', 'EuropeanSevenEdgeCase_l2_4.0'): {'label': 'Tail', 'marker': get_markers()[0], 'color_index': 1},
            ('None_None_None', 'None_l2_4.0'): {'label': 'Prototype', 'marker': get_markers()[1], 'color_index': 2}
        }
    elif model == 'CIFAR10':
        config = {
            "total_weights": 273066,
            "p_v": 0.005
        }

    labels = {
        'FEMNIST': 'FMN-P',
        'CIFAR10': 'C10-P'
    }

    markevery = 30 if model == 'FEMNIST' else 5
    window_size = 1

    setup_plt(square=True)

    with PdfPages(f"{output_dir}/{name}.pdf") as pdf:

        fig, ax = plt.subplots()
        colors, linestyles = get_colorful_styles()

        for i, dataset in enumerate(['FEMNIST', 'CIFAR10']):
            sub_df = df[df["dataset"] == dataset]
            plt.plot(sub_df["keep_number_of_weights"], sub_df["adv_success"], label=labels[dataset],
                     marker=get_markers()[i], color=colors[i + 1], linewidth=2)

        plt.axvline(config["p_v"], color=COLOR_GRAY, linestyle='dashed')

        # custom_lines_colors = []
        # custom_lines_colors_names = []
        # for index, (suf_unprotected, suf_defense) in enumerate(configs.keys()):
        #     values_defended = df[f"adv_success_{suf_defense}"]
        #     values_no_defense = df[f"adv_success_{suf_unprotected}"]
        #     labels = df[f"round"]
        #     config = configs[(suf_unprotected, suf_defense)]
        #
        #     plt.plot(labels, values_defended.rolling(window_size).mean(),
        #              linestyle='dotted', label=config['label'], color=colors[config['color_index']],
        #              linewidth=2, marker=config['marker'], markevery=markevery)
        #     plt.plot(labels, values_no_defense.rolling(window_size).mean(),
        #              color=colors[config['color_index']],
        #              linewidth=2, marker=config['marker'], markevery=markevery)
        #     custom_lines_colors.append(Line2D([0], [0], linestyle="-", lw=2, marker=config['marker'], color=colors[config['color_index']]))
        #     custom_lines_colors_names.append(config['label'])

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

        def percentage_weights_formatter(x, p):
            perc = round(float(x) * 100)
            return f"{perc}%"

        ax.set_xlim(-0.005, 0.1)
        ax.get_xaxis().set_major_formatter(
            matplotlib.ticker.FuncFormatter(percentage_weights_formatter))

        if leftmost:
            plt.ylabel('Malicious Accuracy')
        plt.xlabel('Percentage of weights scaled ($K$)')

        legend_properties = {'weight': 'bold'}
        plt.legend(prop=legend_properties)
        # Legend
        # if leftmost:
        # line = Line2D([0], [0])
        # line.set_visible(False)
        # custom_lines_colors = [line] + custom_lines_colors
        # custom_lines_colors_names = ['Att. Obj:'] + custom_lines_colors_names
        #
        # leg1 = plt.legend(custom_lines_colors, custom_lines_colors_names,
        #                   bbox_to_anchor=(1.02, 1.), loc=4, ncol=6, columnspacing=0.75)
        # ax.add_artist(leg1)
        #
        # # if leftmost:
        # for vpack in leg1._legend_handle_box.get_children()[:1]:
        #     for hpack in vpack.get_children():
        #         del hpack._children[0]
        #
        # if leftmost:
        #     custom_lines_styles = [Line2D([0], [0], linestyle="-", lw=2, color=COLOR_GRAY),
        #                            Line2D([0], [0], linestyle=":", lw=2, color=COLOR_GRAY)]
        #     leg_task = plt.legend(custom_lines_styles, ["No defense", "Norm bound"],
        #                           bbox_to_anchor=(1., 1.), loc=1, ncol=1,
        #                           columnspacing=0.75)
        #     ax.add_artist(leg_task)

        pdf.savefig(bbox_inches='tight', pad_inches=0)
        plt.savefig(f"{output_dir}/{name}.png", bbox_inches='tight', pad_inches=0)
        plt.close()

    return fig, df
