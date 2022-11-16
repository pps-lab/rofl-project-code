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


def load_data(model, bound_type="median"):

    if model == 'FEMNIST':
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
    elif model == 'CIFAR10':
        # query = {
        #     "meta.description": "CIFAR10 increase attackers and blackbox, but temp",
        #     "metrics.result": {'$exists': False}
        # }

        query = {
            "meta.description": "CIFAR10 increase attackers median 1.5"
        }

    docs_list = query_data(query)
    assert len(docs_list) > 0, "Database query is empty!"

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
    if model == 'FEMNIST':
        df = df[df["round"] <= 520]

    elif model == 'CIFAR10':
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

def build_plot(name, df, model, configs=None, leftmost=False):

    if configs is None:
        if model == 'FEMNIST':
            configs = {
                '1_pgd': {'label': '1 (3.3%)', 'marker': get_markers()[0]},
                '2_pgd': {'label': '2 (6.6%)', 'marker': get_markers()[1]},
                '5_pgd': {'label': '5 (16.7%)', 'marker': get_markers()[2]},
                '10_pgd': {'label': '10 (30%)', 'marker': get_markers()[3]}
            }
        elif model == 'CIFAR10':
            # configs = {
            #     '1_pgd': {'label': '1 (2.5%)', 'marker': None},
            #     '2_pgd': {'label': '2 (5%)', 'marker': 's'},
            #     # '5': {'label': '5 (12.5%)', 'marker': 'o'},
            #     '4_pgd': {'label': '4 (10%)', 'marker': 'o'},
            #     '20_pgd': {'label': '20 (50%)', 'marker': 'v'},
            #
            #     # '25_pgd': {'label': '25 (62.5%)', 'marker': 'v'}
            # }
            configs = {
                f"{num}_pgd": {'label': f"{num} ({num / 0.4}%)", 'marker': get_markers()[i]} for i, num in enumerate([1, 2, 5, 10, 20])
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
            values_acc = df[f"accuracy_{suffix}"]
            values_adv = df[f"adv_success_{suffix}"]
            labels = df[f"round"]
            config = configs[suffix]

            plt.plot(labels, values_acc.rolling(window_size).mean().shift(-window_size),
                     linestyle='dotted', label=config['label'], color=config['color'],
                     linewidth=2, marker=config['marker'], markevery=markevery)
            plt.plot(labels, values_adv.rolling(window_size).mean().shift(-window_size),
                     color=config['color'],
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
        elif model == 'CIFAR10':
            custom_lines_colors.insert(3, line)
            custom_lines_colors_names.insert(3, '')
            ordering = [0, 3, 1, 4, 2, 5]

        leg1 = plt.legend(itemgetter(*ordering)(custom_lines_colors), itemgetter(*ordering)(custom_lines_colors_names),
                          bbox_to_anchor=(1.02, 1.), loc=4, ncol=3, columnspacing=0.75)
        ax.add_artist(leg1)

        # if leftmost:
        for vpack in leg1._legend_handle_box.get_children()[:1]:
            for hpack in vpack.get_children():
                del hpack._children[0]

        if leftmost:
            custom_lines_styles = [Line2D([0], [0], linestyle="-", lw=2, color=COLOR_GRAY),
                                   Line2D([0], [0], linestyle=":", lw=2, color=COLOR_GRAY)]
            leg_task = plt.legend(custom_lines_styles, ["Backdoor task", "Main task"],
                                  bbox_to_anchor=(0., 0.53), loc='lower left', ncol=1,
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
                          bbox_to_anchor=(1., 0.28, 1., .102), loc=3, ncol=1, columnspacing=0.75, title="Attackers")
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
