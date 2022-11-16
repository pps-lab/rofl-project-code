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
                # { "meta.description": "FEMNIST edgecase m15 anticipate"}
                { "meta.description": "FEMNIST edgecase pixel"}
            ]
        }
    else:
        query = {
            "$or": [
                { "meta.description": "FEMNIST show comparison of pgd and blackbox for edge cases and non edge cases", },
                { "meta.description": "FEMNIST edgecase anticipate test", }
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
        pixel_position = get_pixel_position(metric)
        metrics[i] = _preprocess(df, f"{backdoor_type}_{edge_case_p}_{pixel_position}_{clip_type}_{attack_type}{'_noattack' if not_attacking else ''}")

    df = reduce(lambda left, right: pd.merge(left, right, on=['round'], how='outer'), metrics)
    df = df[df["round"] <= 520]
    return df

def get_pixel_position(metric):
    if 'trigger_position' in metric['client']['malicious']['backdoor']:
        return metric['client']['malicious']['backdoor']['trigger_position']
    else:
        return None

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
            {"meta.description": "CIFAR edge case compare anticipate temp"},
            {"meta.description": "CIFAR10 pixel"}
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
            if 'backdoor' in metric['hyperparameters']['args']['client']['malicious'] and \
               'edge_case_p' in metric['hyperparameters']['args']['client']['malicious']['backdoor'] is not None else None
        clip_type = metric['client']['clip']['type'] \
            if 'clip' in metric['client'] and \
               metric['client']['clip'] is not None else None
        pixel_position = get_pixel_position(metric)
        print(f"{backdoor_type}_{edge_case_p}_{pixel_position}_{clip_type}_{attack_type}{'_noattack' if not_attacking else ''}")
        metrics[i] = _preprocess(df, f"{backdoor_type}_{edge_case_p}_{pixel_position}_{clip_type}_{attack_type}{'_noattack' if not_attacking else ''}")

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


def build_attack_plot(name, df, model, configs=None, leftmost=False, grid=True, noise=True, animation_step=3):

    colors = get_colors_attack_objective()
    markers = get_markers()

    if model == 'FEMNIST_median':
        configs = {
            'FEMNISTRandomNoiseEdgeCase_None_None_median_l2': {'label': 'Noise', 'marker': markers[0], 'color': colors[0]},
            'EuropeanSevenEdgeCase_0.9_None_median_l2': {'label': 'Tail', 'marker': markers[1], 'color': colors[1]},
            'tasks_pixel_pattern_None_10.0_median_l2': {'label': 'Prototype + Pixel', 'marker': markers[2], 'color': colors[2]},
            'tasks_None_None_median_l2': {'label': 'Prototype', 'marker': markers[2], 'color': colors[2]},
            # 'pixel_pattern_None_median_l2': {'label': 'Pixel-pattern', 'marker': get_markers()[3], 'color_index': 4},
        }
    elif model == 'CIFAR10':
        if noise:
            configs = {
                # 'tasks_None_None': {'label': 'No norm bound', 'marker': None, 'color_index': 0},
                'CifarRandomNoiseEdgeCase_None_None_median_l2': {'label': 'Noise', 'marker': markers[0], 'color': colors[0]},
                'NorthWesternEdgeCase_None_None_median_l2': {'label': 'Tail', 'marker': markers[1], 'color': colors[1]},
                'semantic_pixel_pattern_None_4.0_median_l2': {'label': 'Prototype + Pixel', 'marker': markers[2], 'color': colors[2]},
                'semantic_None_None_median_l2': {'label': 'Prototype', 'marker': markers[2], 'color': colors[2]},
                # 'pixel_pattern_median_l2': {'label': 'Pixel-pattern', 'marker': get_markers()[2], 'color_index': 3},

                # 'tasks_blackbox_None_l2': {'label': 'Blackbox (Handwriters)', 'marker': 's', 'color_index': 2},
                # 'edge_blackbox_0.9_l2': {'label': 'Blackbox (Edge-case)', 'marker': 'v', 'color_index': 2}
            }
        else:
            if animation_step == 1:
                configs = {
                    'semantic_median_l2': {'label': 'Prototype', 'marker': get_markers()[2], 'color_index': 2},
                }
            else:
                configs = {
                    'NorthWesternEdgeCase_median_l2': {'label': 'Tail', 'marker': get_markers()[1], 'color_index': 1},
                    'semantic_median_l2': {'label': 'Prototype', 'marker': get_markers()[2], 'color_index': 2},
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
            # values_blackbox = df[f"adv_success_{suffix}_None_blackbox"]

            labels = df[f"round"]
            config = configs[suffix]

            linestyle = LINESTYLE_DP if 'tasks_pixel_pattern' in suffix else LINESTYLE_PGD

            plt.plot(labels, values_pgd.rolling(window_size).mean().shift(-window_size),
                     linestyle=linestyle, label=config['label'], color=config['color'],
                     linewidth=2, marker=config['marker'], markevery=markevery)

            # plt.plot(labels, values_blackbox.rolling(window_size).mean().shift(-window_size),
            #          linestyle=LINESTYLE_DP,
            #          color=config['color'],
            #          linewidth=2, marker=config['marker'], markevery=markevery, path_effects=[DP_PATHEFFECT])

            # values_anticipate = df[f"adv_success_{suffix}_l2_anticipate"]
            # plt.plot(labels, values_anticipate.rolling(window_size).mean().shift(-window_size),
            #          linestyle=LINESTYLE_AT, label=config['label'], color=config['color'],
            #          linewidth=2, marker=config['marker'], markevery=markevery)
            # values_neurotoxin = df[f"adv_success_{suffix}_l2_neurotoxin"]
            # plt.plot(labels, values_neurotoxin.rolling(window_size).mean().shift(-window_size),
            #          linestyle=LINESTYLE_NT, label=config['label'], color=config['color'],
            #          linewidth=2, marker=config['marker'], markevery=markevery)

            custom_lines_colors.append(Line2D([0], [0], linestyle=linestyle, lw=2, marker=config['marker'], color=config['color']))
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
                              bbox_to_anchor=(-0.025, 1.), loc='lower left', ncol=6, columnspacing=0.75)
            #1.16 bbox anchor
            ax.add_artist(leg1)

            # if leftmost:
            for vpack in leg1._legend_handle_box.get_children()[:1]:
                for hpack in vpack.get_children():
                    del hpack._children[0]

        # if leftmost:
        #     line = Line2D([0], [0])
        #     line.set_visible(False)
        #     custom_lines_styles = [
        #         line,
        #         Line2D([0], [0], linestyle=LINESTYLE_AT, lw=2, color=COLOR_GRAY),
        #         Line2D([0], [0], linestyle=LINESTYLE_PGD, lw=2, color=COLOR_GRAY),
        #         Line2D([0], [0], linestyle=LINESTYLE_NT, lw=2, color=COLOR_GRAY),
        #         Line2D([0], [0], linestyle=LINESTYLE_DP, lw=2, color=COLOR_GRAY, path_effects=[DP_PATHEFFECT])]
        #     leg_task = plt.legend(custom_lines_styles, ["   Attack:", LABEL_AT, LABEL_PGD, LABEL_NT, LABEL_DP],
        #                           bbox_to_anchor=(-0.025, 0.99), loc='lower left', ncol=6,
        #                           columnspacing=0.75)
        #     for vpack in leg_task._legend_handle_box.get_children()[:1]:
        #         for hpack in vpack.get_children():
        #             del hpack._children[0]
        #     ax.add_artist(leg_task)

        # plt.legend(title='Bound', mode="expand", loc="lower left", labelspacing=.05, bbox_to_anchor=(1.01, 0, .6, 0))

        pdf.savefig(bbox_inches='tight', pad_inches=0)
        plt.savefig(f"{output_dir}/{name}.png", bbox_inches='tight', pad_inches=0)
        plt.close()

    return fig, df
