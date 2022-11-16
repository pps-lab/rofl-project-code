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

from common import _preprocess, setup_plt, query_data, get_colorful_styles, output_dir, COLOR_GRAY, COLOR_NO_BOUND, yaxis_formatter, get_markers, get_colors_bounds
from common import LABEL_AT, LABEL_PGD, LABEL_NT, LABEL_DP

def load_data_femnist():

    query = {
        # "meta.description": "FEMNIST single-shot under bound (and edge case)",
        # "hyperparameters.args.client.malicious.backdoor.type": {'$ne': 'edge'}
        "$or": [
            {"meta.description": "FEMNIST single shot new attacks"},
            {"meta.description": "FEMNIST single shot anticipate"},
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
        attack_method  = get_attack_method(metric)
        backdoor_type = get_backdoor_type(metric)
        clip_type = metric['hyperparameters']['args']['client']['clip']['type'] \
                            if metric['hyperparameters']['args']['client']['clip'] is not None else None
        bound = metric['hyperparameters']['args']['client']['clip']['value'] \
                            if metric['hyperparameters']['args']['client']['clip'] is not None else None
        # pgd = metric['hyperparameters']['args']['client']['malicious']['evasion']['args']['pgd_factor'] \
        #                     if 'malicious' in metric['hyperparameters']['args']['client'] and \
        #                        'evasion' in metric['hyperparameters']['args']['client']['malicious'] and \
        #                        'pgd_factor' in metric['hyperparameters']['args']['client']['malicious']['evasion']['args'] else None
        metrics[i] = _preprocess(df, f"{backdoor_type}_{clip_type}_{bound}_{attack_method}")

    df = reduce(lambda left, right: pd.merge(left, right, on=['round'], how='outer'), metrics)
    df = df[df["round"] <= 15]
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
    if 'attack_stop' in metric['client']['malicious'] and metric['client']['malicious']['attack_stop'] == 0:
        return 'pgd_noattack'
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

def load_data_cifar():
    query = {
        "$or": [
            {"meta.description": "CIFAR10 single shot comparison of edge cases and defenses"},
            {"meta.description": "CIFAR10 single shot new attacks"}
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
        backdoor_type = get_backdoor_type(metric)
        attack_method = get_attack_method(metric)
        clip_type = metric['hyperparameters']['args']['client']['clip']['type'] \
            if 'clip' in metric['hyperparameters']['args']['client'] and \
               metric['hyperparameters']['args']['client']['clip'] is not None else None
        bound = metric['hyperparameters']['args']['client']['clip']['value'] \
            if 'clip' in metric['hyperparameters']['args']['client'] is not None else None
        metrics[i] = _preprocess(df, f"{backdoor_type}_{clip_type}_{bound}_{attack_method}")

    df = reduce(lambda left, right: pd.merge(left, right, on=['round'], how='outer'), metrics)
    df = df[df["round"] <= 15]
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



def build_plot(name, df, model):

    if model == 'FEMNIST':
        # configs = {
        #     'None_None_None': {'label': 'None', 'marker': None},
        #     'l2_4.0_0.16667': {'label': '$L_2$ (4.0)', 'marker': 's'},
        #     'median_l2_2.0_2.0': {'label': 'Median (2.0)', 'marker': 'o'},
        # }
        configs = {
            'None_None_None_pgd': {'label': 'None', 'marker': None},
            'l2_4.0_0.16667_pgd': {'label': '$L_2$ (4.0)', 'marker': 's'},
        }
    elif model == 'CIFAR10':
        configs = {
            'semantic_None_None': {'label': 'None', 'marker': None},
            'semantic_l2_5.0': {'label': '$L_2$ (5.0)', 'marker': 's'},
        }
    # elif model == 'CIFAR10_attacks':
    #     configs = {
    #         'bgwall_None_None': {'label': 'A1-WALL', 'marker': 's'},
    #         'greencar_None_None': {'label': 'A2-GREEN', 'marker': 'o'},
    #         'racingstripe_None_None': {'label': 'A3-STRIPE', 'marker': 'v'}
    #     }
    # elif model == 'CIFAR10_defenses':
    #     configs = {
    #         'bgwall_None_None': {'label': 'None', 'marker': None},
    #         'bgwall_l2_5.0': {'label': '$L_2$ (5.0)', 'marker': 's'},
    #         'bgwall_median_l2_2.0': {'label': 'Median (2.0)', 'marker': 'o'},
    #     }


    markevery = 100
    window_size = 1

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
        ax.set_ylim(ymin=-0.05, ymax=1.01)
        ax.set_ylabel("Accuracy ($\\%$)")
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])

        plt.ylabel('Accuracy ($\\%$)')
        plt.xlabel('Round')

        # Legend
        custom_lines_styles = [Line2D([0], [0], linestyle="-", lw=2, color=COLOR_GRAY),
                               Line2D([0], [0], linestyle=":", lw=2, color=COLOR_GRAY)]
        leg1 = plt.legend(custom_lines_colors, custom_lines_colors_names,
                          bbox_to_anchor=(1., 0.28, 1., .102), loc=3, ncol=1, columnspacing=0.75, title="Bound")
        leg1._legend_box.align = "left"

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

def build_plot_tight(name, df, model, leftmost):
    bound_colors = get_colors_bounds()

    alpha_seethrough = 1.0
    if model == 'FEMNIST':
        # configs = {
        #     'None_None_None': {'label': 'None', 'marker': get_markers()[0]},
        #     'l2_4.0_0.16667': {'label': '4.0', 'marker': get_markers()[1]}, # 4.0
        #     # 'median_l2_2.0_2.0': {'label': 'Median (2.0)', 'marker': 'o'},
        # }
        configs = {
            'None_l2_4.0_anticipate': {'label': f'Bound ({LABEL_AT})', 'marker': get_markers()[0], 'color': bound_colors[0], 'alpha': 1.0},
            'None_l2_4.0_pgd': {'label': f'Bound ({LABEL_PGD})', 'marker': get_markers()[1], 'color': bound_colors[0], 'alpha': alpha_seethrough},
            'None_l2_4.0_neurotoxin': {'label': f'Bound ({LABEL_NT})', 'marker': get_markers()[2], 'color': bound_colors[0], 'alpha': alpha_seethrough},
            'None_None_None_pgd': {'label': 'No Bound', 'marker': None, 'color': COLOR_NO_BOUND, 'alpha': 1.0},
        }
    elif model == 'CIFAR10':
        configs = {
            'semantic_l2_5.0_anticipate': {'label': f'Bound ({LABEL_AT})', 'marker': get_markers()[0], 'color': bound_colors[0], 'alpha': 1.0}, # 5.0
            'semantic_l2_5.0_pgd': {'label': f'Bound ({LABEL_PGD})', 'marker': get_markers()[1], 'color': bound_colors[0], 'alpha': alpha_seethrough}, # 5.0
            'semantic_l2_5.0_neurotoxin': {'label': f'Bound ({LABEL_NT})', 'marker': get_markers()[2], 'color': bound_colors[0], 'alpha': alpha_seethrough}, # 5.0
            'semantic_None_None_pgd': {'label': 'No Bound', 'marker': None, 'color': COLOR_NO_BOUND, 'alpha': 1.0},
        }

    markevery = 1
    window_size = 1

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

            # if config['interleave_offset'] is not None:
            #     offset = config['interleave_offset']
            #     idx_to_keep = set(range(offset, len(values_acc), 3))
            #     idx_to_remove = list(set(range(len(values_acc))) - idx_to_keep)
            #     # print(idx_to_remove)
            #     values_acc = values_acc.drop(idx_to_remove)
            #     values_adv = values_adv.drop(idx_to_remove)
            #     labels = labels.drop(idx_to_remove)

            plt.plot(labels, values_acc,
                     linestyle='dotted', label=config['label'], color=config['color'],
                     linewidth=2, marker=config['marker'], markevery=markevery, alpha=config['alpha'])
            plt.plot(labels, values_adv,
                     color=config['color'],
                     linewidth=2, marker=config['marker'], markevery=markevery, alpha=config['alpha'])
            custom_lines_colors.append(Line2D([0], [0], linestyle="-", lw=2, marker=config['marker'], color=config['color'], alpha=config['alpha']))
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

        # line = Line2D([0], [0])
        # line.set_visible(False)
        # custom_lines_colors = [line] + custom_lines_colors
        # custom_lines_colors_names = ['$L_2$-B:'] + custom_lines_colors_names

        if leftmost:
            leg1 = plt.legend(custom_lines_colors, custom_lines_colors_names,
                              bbox_to_anchor=(-0.025, 1.), loc='lower left', ncol=6, columnspacing=0.75)
            ax.add_artist(leg1)

        # if leftmost:
        # for vpack in leg1._legend_handle_box.get_children()[:1]:
        #     for hpack in vpack.get_children():
        #         del hpack._children[0]

        if leftmost:
            custom_lines_styles = [Line2D([0], [0], linestyle="-", lw=2, color=COLOR_GRAY),
                                   Line2D([0], [0], linestyle=":", lw=2, color=COLOR_GRAY)]
            leg_task = plt.legend(custom_lines_styles, ["Backdoor task", "Main task"],
                                  bbox_to_anchor=(1., 0.07), loc=4, ncol=1,
                                  columnspacing=0.75)
            ax.add_artist(leg_task)

        # ax.set_ylim(ymin=-0.05, ymax=1.05)
        # ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        # if model != "CIFAR10":
        #     ax.set_ylabel("Accuracy ($\\%$)")
        # else:
        #     ax.set_ylabel(" ")
        #
        # plt.xlabel('Round')
        #
        # if show_legend:
        #     # Legend
        #     custom_lines_styles = [Line2D([0], [0], linestyle="-", lw=2, color=COLOR_GRAY),
        #                            Line2D([0], [0], linestyle=":", lw=2, color=COLOR_GRAY)]
        #     leg1 = plt.legend(custom_lines_colors, custom_lines_colors_names, loc=3, bbox_to_anchor=(0.0, 0.37), ncol=1, columnspacing=0.75)
        #
        #     leg2 = plt.legend(custom_lines_styles, ["Backdoor task", "Main task"],
        #                       loc=3, bbox_to_anchor=(0.0, 0.05), ncol=1,
        #                       columnspacing=0.75)
        #     ax.add_artist(leg1)
        #     ax.add_artist(leg2)

        # plt.legend(title='Bound', mode="expand", loc="lower left", labelspacing=.05, bbox_to_anchor=(1.01, 0, .6, 0))

        pdf.savefig(bbox_inches='tight', pad_inches=0)
        plt.savefig(f"{output_dir}/{name}.png", bbox_inches='tight', pad_inches=0)
        plt.close()

    return fig, df


def build_plot_presentation(name, df, model, leftmost, animation_step):

    if animation_step == 1:
        configs = {
            'None_None_None': {'label': 'None', 'marker': get_markers()[0]},
            # 'l2_4.0_0.16667': {'label': '4.0', 'marker': get_markers()[1]}, # 4.0
            # 'median_l2_2.0_2.0': {'label': 'Median (2.0)', 'marker': 'o'},
        }
    elif animation_step == 2:
        configs = {
            'None_None_None': {'label': 'None', 'marker': get_markers()[0]},
            'l2_4.0_0.16667': {'label': '4.0', 'marker': get_markers()[1]}, # 4.0
            # 'median_l2_2.0_2.0': {'label': 'Median (2.0)', 'marker': 'o'},
        }
    else:
        raise ValueError(f"Animation step {animation_step} not supported!")

    markevery = 1
    window_size = 1

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
        # ax.grid(True, axis="y", linestyle=':', color='0.6', zorder=0, linewidth=1.2)
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
        custom_lines_colors_names = ['$L_2$-Bound:'] + custom_lines_colors_names

        leg1 = plt.legend(custom_lines_colors, custom_lines_colors_names,
                          bbox_to_anchor=(1.02, 1.), loc=4, ncol=6, columnspacing=0.75)
        ax.add_artist(leg1)

        # if leftmost:
        for vpack in leg1._legend_handle_box.get_children()[:1]:
            for hpack in vpack.get_children():
                del hpack._children[0]

        if leftmost:
            custom_lines_styles = [Line2D([0], [0], linestyle="-", lw=2, color=COLOR_GRAY),
                                   Line2D([0], [0], linestyle=":", lw=2, color=COLOR_GRAY)]
            leg_task = plt.legend(custom_lines_styles[::-1], ["Backdoor task", "Main task"][::-1],
                                  bbox_to_anchor=(1., 0.07), loc=4, ncol=1,
                                  columnspacing=0.75)
            ax.add_artist(leg_task)

        # ax.set_ylim(ymin=-0.05, ymax=1.05)
        # ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        # if model != "CIFAR10":
        #     ax.set_ylabel("Accuracy ($\\%$)")
        # else:
        #     ax.set_ylabel(" ")
        #
        # plt.xlabel('Round')
        #
        # if show_legend:
        #     # Legend
        #     custom_lines_styles = [Line2D([0], [0], linestyle="-", lw=2, color=COLOR_GRAY),
        #                            Line2D([0], [0], linestyle=":", lw=2, color=COLOR_GRAY)]
        #     leg1 = plt.legend(custom_lines_colors, custom_lines_colors_names, loc=3, bbox_to_anchor=(0.0, 0.37), ncol=1, columnspacing=0.75)
        #
        #     leg2 = plt.legend(custom_lines_styles, ["Backdoor task", "Main task"],
        #                       loc=3, bbox_to_anchor=(0.0, 0.05), ncol=1,
        #                       columnspacing=0.75)
        #     ax.add_artist(leg1)
        #     ax.add_artist(leg2)

        # plt.legend(title='Bound', mode="expand", loc="lower left", labelspacing=.05, bbox_to_anchor=(1.01, 0, .6, 0))

        pdf.savefig(bbox_inches='tight', pad_inches=0)
        plt.savefig(f"{output_dir}/{name}.png", bbox_inches='tight', pad_inches=0)
        plt.close()

    return fig, df
