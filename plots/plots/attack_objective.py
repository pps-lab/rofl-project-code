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

from common import _preprocess, setup_plt, query_data, get_colorful_styles, output_dir, COLOR_GRAY, yaxis_formatter, get_markers, get_colors_attack_objective
from common import LINESTYLE_AT, LINESTYLE_PGD, LINESTYLE_NT, LINESTYLE_DP, DP_PATHEFFECT
from common import LABEL_AT, LABEL_PGD, LABEL_NT, LABEL_DP

def load_data_femnist(type="l2"):

    if type == "median":
        query = {
            "$or": [
                # { "meta.description": "FEMNIST show comparison of pgd and blackbox for edge cases and non edge cases (median)", },
                # { "meta.description": "FEMNIST edgecase anticipate test", },
                # { "meta.description": "FEMNIST edgecase full" },
                { "meta.description": "FEMNIST edgecase m15"},
                { "meta.description": "FEMNIST edgecase m15 anticipate"}
            ]
        }
    else:
        query = {
            "$or": [
                { "meta.description": "FEMNIST show comparison of pgd and blackbox for edge cases and non edge cases" },
                { "meta.description": "FEMNIST edgecase anticipate test" }
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
        backdoor_type = get_backdoor_type(metric)
        attack_type = get_attack_method(metric)
        edge_case_p = metric['hyperparameters']['args']['client']['malicious']['backdoor']['edge_case_p'] \
                            if 'backdoor' in metric['hyperparameters']['args']['client']['malicious'] and\
                               'edge_case_p' in metric['hyperparameters']['args']['client']['malicious']['backdoor'] is not None else None
        clip_type = metric['client']['clip']['type'] \
                            if 'clip' in metric['client'] and \
                               metric['client']['clip'] is not None else None
        metrics[i] = _preprocess(df, f"{backdoor_type}_{edge_case_p}_{clip_type}_{attack_type}{'_noattack' if not_attacking else ''}")

    df = reduce(lambda left, right: pd.merge(left, right, on=['round'], how='outer'), metrics)
    df = df[df["round"] <= 520]
    return df

def get_attack_method(metric):
    if metric['client']['malicious']['evasion'] == None:
        return 'None_blackbox'
    elif metric['client']['malicious']['objective']['name'] == "TargetedAttack":
        if metric['client']['malicious']['evasion']['name'] == "NeurotoxinEvasion":
            return 'l2_neurotoxin'
        else:
            return 'median_l2_pgd'
    elif metric['client']['malicious']['objective']['name'] == "AnticipateTfAttack":
        return 'l2_anticipate'
    else:
        return 'ERROR'

def load_data_cifar():
    query = {
        "$or": [
            {"meta.description": "CIFAR10 edge case and prototype"},
            {"meta.description": "CIFAR edgecase neurotoxin"},
            # {"meta.description": "CIFAR edgecase anticipate"},
            # {"meta.description": "CIFAR10 anticipate edge"},
            {"meta.description": "CIFAR edge case compare anticipate temp"}
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
        backdoor_type = get_backdoor_type(metric)
        evasion_type = get_evasion_type(metric)
        attack_type = get_attack_method_cifar(metric)
        clip_type = metric['client']['clip']['type'] \
                            if 'clip' in metric['client'] and \
                               metric['client']['clip'] is not None else None
        metrics[i] = _preprocess(df, f"{backdoor_type}_{clip_type}_{evasion_type}_{attack_type}{'_noattack' if not_attacking else ''}")

    df = reduce(lambda left, right: pd.merge(left, right, on=['round'], how='outer'), metrics)
    df = df[df["round"] <= 520]
    return df

def get_attack_method_cifar(metric):
    if metric['client']['malicious']['evasion'] == None:
        return 'blackbox'
    elif metric['client']['malicious']['objective']['name'] == "TargetedAttack":
        if metric['client']['malicious']['evasion']['name'] == "NeurotoxinEvasion":
            return 'neurotoxin'
        else:
            return 'pgd'
    elif metric['client']['malicious']['objective']['name'] == "AnticipateAttack" or \
        metric['client']['malicious']['objective']['name'] == "AnticipateTfAttack":
        return 'anticipate'
    else:
        return 'ERROR'

def get_backdoor_type(metric):
    if 'backdoor' not in metric['hyperparameters']['args']['client']['malicious']:
        return None
    elif metric['hyperparameters']['args']['client']['malicious']['backdoor']['type'] == 'edge':
        return metric['hyperparameters']['args']['client']['malicious']['backdoor']['edge_case_type']
    else:
        return metric['hyperparameters']['args']['client']['malicious']['backdoor']['type']

def get_evasion_type(metric):
    if 'evasion' in metric['hyperparameters']['args']['client']['malicious']:
        return metric['hyperparameters']['args']['client']['malicious']['evasion']['args']['norm_type']
    else:
        return None


def build_plot(name, df, model, configs=None):

    if configs is None:
        if model == 'FEMNIST_edge':
            configs = {
                'edge_0.9_None_median_l2_pgd': {'label': 'No norm bound', 'marker': None, 'color_index': 0},
                'edge_0.9_l2_median_l2_pgd': {'label': 'PGD (Tail)', 'marker': 'v', 'color_index': 1},
                'edge_0.9_l2_None_blackbox': {'label': 'Blackbox (Tail)', 'marker': 'v', 'color_index': 2}
            }
        elif model == 'FEMNIST_tasks':
            configs = {
                'tasks_None_None_median_l2_pgd': {'label': 'No norm bound', 'marker': None, 'color_index': 0},
                'tasks_None_l2_median_l2_pgd': {'label': 'PGD (Handwriters)', 'marker': 's', 'color_index': 1},
                'tasks_None_l2_None_blackbox': {'label': 'Blackbox (Handwriters)', 'marker': 's', 'color_index': 2}
            }
        elif model == 'CIFAR10':
            configs = {
                'bgwall': {'label': 'A1-WALL', 'marker': 's'},
                'greencar': {'label': 'A2-GREEN', 'marker': 'o'},
                'racingstripe': {'label': 'A3-STRIPE', 'marker': 'v'}
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
        ax.set_ylim(ymin=-0.01, ymax=1.01)
        ax.set_ylabel("Accuracy ($\\%$)")
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])

        plt.ylabel('Accuracy ($\\%$)')
        plt.xlabel('Round')

        # Legend
        custom_lines_styles = [Line2D([0], [0], linestyle="-", lw=2, color=COLOR_GRAY),
                               Line2D([0], [0], linestyle=":", lw=2, color=COLOR_GRAY)]
        leg1 = plt.legend(custom_lines_colors, custom_lines_colors_names,
                          bbox_to_anchor=(1., 0.28, 1., .102), loc=3, ncol=1, columnspacing=0.75, title="Training method")
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


def build_attack_plot(name, df, model, configs=None, leftmost=False, grid=True, noise=True, animation_step=3):

    # if configs is None:
    #     if model == 'FEMNIST':
    #         configs = {
    #             'edge_pgd_0.9_None': {'label': 'No norm bound', 'marker': None, 'color_index': 0},
    #             'edge_pgd_0.9_l2': {'label': 'PGD (Edge-case)', 'marker': 'v', 'color_index': 1},
    #             'edge_blackbox_0.9_l2': {'label': 'Blackbox (Edge-case)', 'marker': 'v', 'color_index': 2}
    #         }
    #     elif model == 'FEMNIST_tasks':
    #         configs = {
    #             'tasks_pgd_None_None': {'label': 'No norm bound', 'marker': None, 'color_index': 0},
    #             'tasks_pgd_None_l2': {'label': 'PGD (Handwriters)', 'marker': 's', 'color_index': 1},
    #             'tasks_blackbox_None_l2': {'label': 'Blackbox (Handwriters)', 'marker': 's', 'color_index': 2}
    #         }
    #     elif model == 'CIFAR10':
    #         configs = {
    #             'bgwall': {'label': 'A1-WALL', 'marker': 's'},
    #             'greencar': {'label': 'A2-GREEN', 'marker': 'o'},
    #             'racingstripe': {'label': 'A3-STRIPE', 'marker': 'v'}
    #         }

    colors = get_colors_attack_objective()
    markers = get_markers()

    if model == 'FEMNIST_l2':
        configs = {
            'edge_0.9_l2': {'label': 'Tail', 'marker': get_markers()[0], 'color_index': 1},
            'tasks_None_l2': {'label': 'Prototype', 'marker': get_markers()[1], 'color_index': 2},
        }
    elif model == 'FEMNIST_median':
        if noise:
            configs = {
                'FEMNISTRandomNoiseEdgeCase_None_median_l2': {'label': 'Noise', 'marker': markers[0], 'color': colors[0]},
                'EuropeanSevenEdgeCase_0.9_median_l2': {'label': 'Tail', 'marker': markers[1], 'color': colors[1]},
                'tasks_None_median_l2': {'label': 'Prototype', 'marker': markers[2], 'color': colors[2]},
                # 'pixel_pattern_None_median_l2': {'label': 'Pixel-pattern', 'marker': get_markers()[3], 'color_index': 4},
            }
        else:
            if animation_step == 1:
                configs = {
                    'tasks_None_median_l2': {'label': 'Prototype', 'marker': markers[2], 'color': colors[2]},
                }
            elif animation_step == 2:
                configs = {
                    'EuropeanSevenEdgeCase_0.9_median_l2': {'label': 'Tail', 'marker': markers[1], 'color': colors[1]},
                    'tasks_None_median_l2': {'label': 'Prototype', 'marker': markers[2], 'color': colors[2]},
                }

    elif model == 'CIFAR10':
        if noise:
            configs = {
                # 'tasks_None_None': {'label': 'No norm bound', 'marker': None, 'color_index': 0},
                'CifarRandomNoiseEdgeCase_median_l2': {'label': 'Noise', 'marker': markers[0], 'color': colors[0]},
                'NorthWesternEdgeCase_median_l2': {'label': 'Tail', 'marker': markers[1], 'color': colors[1]},
                'semantic_median_l2': {'label': 'Prototype', 'marker': markers[2], 'color': colors[2]},
                # 'pixel_pattern_median_l2': {'label': 'Pixel-pattern', 'marker': get_markers()[2], 'color_index': 3},

                # 'tasks_blackbox_None_l2': {'label': 'Blackbox (Handwriters)', 'marker': 's', 'color_index': 2},
                # 'edge_blackbox_0.9_l2': {'label': 'Blackbox (Edge-case)', 'marker': 'v', 'color_index': 2}
            }
        else:
            if animation_step == 1:
                configs = {
                    'semantic_median_l2': {'label': 'Prototype', 'marker': markers[2], 'color': colors[2]},
                }
            else:
                configs = {
                    'NorthWesternEdgeCase_median_l2': {'label': 'Tail', 'marker': markers[1], 'color': colors[1]},
                    'semantic_median_l2': {'label': 'Prototype', 'marker': markers[2], 'color': colors[2]},
                }
        df = df[df["round"] <= 800]

    markevery = 100
    window_size = 20

    setup_plt(square=True)

    with PdfPages(f"{output_dir}/{name}.pdf") as pdf:

        fig, ax = plt.subplots()

        custom_lines_colors = []
        custom_lines_colors_names = []
        for index, suffix in enumerate(configs.keys()):
            values_pgd = df[f"adv_success_{suffix}_median_l2_pgd"]
            values_blackbox = df[f"adv_success_{suffix}_None_blackbox"]

            labels = df[f"round"]
            config = configs[suffix]

            plt.plot(labels, values_pgd.rolling(window_size).mean().shift(-window_size),
                     linestyle=LINESTYLE_PGD, label=config['label'], color=config['color'],
                     linewidth=2, marker=config['marker'], markevery=markevery)

            plt.plot(labels, values_blackbox.rolling(window_size).mean().shift(-window_size),
                     linestyle=LINESTYLE_DP,
                     color=config['color'],
                     linewidth=2, marker=config['marker'], markevery=markevery)

            values_anticipate = df[f"adv_success_{suffix}_l2_anticipate"]
            plt.plot(labels, values_anticipate.rolling(window_size).mean().shift(-window_size),
                     linestyle=LINESTYLE_AT, label=config['label'], color=config['color'],
                     linewidth=2, marker=config['marker'], markevery=markevery)
            values_neurotoxin = df[f"adv_success_{suffix}_l2_neurotoxin"]
            plt.plot(labels, values_neurotoxin.rolling(window_size).mean().shift(-window_size),
                     linestyle=LINESTYLE_NT, label=config['label'], color=config['color'],
                     linewidth=2, marker=config['marker'], markevery=markevery)

            custom_lines_colors.append(Line2D([0], [0], linestyle="-", lw=2, marker=config['marker'], color=config['color']))
            custom_lines_colors_names.append(config['label'])

        ##########################
        # General Format
        ##########################
        if grid:
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
            custom_lines_colors_names = ['Att. Obj:'] + custom_lines_colors_names

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
                Line2D([0], [0], linestyle=LINESTYLE_DP, lw=2, color=COLOR_GRAY)]
            leg_task = plt.legend(custom_lines_styles, ["   Attack:", LABEL_AT, LABEL_PGD, LABEL_NT, LABEL_DP],
                                  bbox_to_anchor=(-0.025, 0.99), loc='lower left', ncol=6,
                                  columnspacing=0.75)
            for vpack in leg_task._legend_handle_box.get_children()[:1]:
                for hpack in vpack.get_children():
                    del hpack._children[0]
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
