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
import re

def load_continuous_bound_data(model, type):

    if model == 'FEMNIST':
        query = {
            "meta.description": "FEMNIST continuous under different bounds",
            # 'hyperparameters.args.client.malicious.attack_stop': { '$ne': 0 }
        }
    elif model == 'FEMNIST_linf':
        query = {
            "meta.description": "FEMNIST comparison of static bounds for linfty norm",
        }
    elif model == 'CIFAR10':
        # rgx = re.compile('.*cifar_continuous_bound_racingstripes_1618325970_.*', re.IGNORECASE)  # compile the regex
        #
        # query = {
        #     "meta.description": "CIFAR10 Continuous under different static bounds",
        #     "_id": rgx
        # }
        # desc = "CIFAR10 static bounds for green cars"
        desc = "CIFAR10 green car median" if type == "median" else "CIFAR10 static bounds for green cars"
        query = {
            "meta.description": desc
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
        not_attacking = 'attack_stop' in metric['hyperparameters']['args']['client']['malicious'] or \
                        'attack_start' in metric['hyperparameters']['args']['client']['malicious'] \
                        if 'malicious' in metric['hyperparameters']['args']['client'] is not None else None
        clip_type = metric['hyperparameters']['args']['client']['clip']['type'] \
                            if metric['hyperparameters']['args']['client']['clip'] is not None else None
        bound = metric['hyperparameters']['args']['client']['clip']['value'] \
                            if metric['hyperparameters']['args']['client']['clip'] is not None else None
        pgd = metric['hyperparameters']['args']['client']['malicious']['evasion']['args']['pgd_factor'] \
                            if 'malicious' in metric['hyperparameters']['args']['client'] and \
                               'evasion' in metric['hyperparameters']['args']['client']['malicious'] and \
                               metric['hyperparameters']['args']['client']['malicious']['evasion'] is not None and \
                               'pgd_factor' in metric['hyperparameters']['args']['client']['malicious']['evasion']['args'] else None
        attack_method = 'blackbox' if metric['client']['malicious']['objective']['args']['num_epochs'] == 2 else 'pgd'
        metrics_df.append(_preprocess(df, f"{clip_type}_{bound}_{pgd}_{attack_method}{'_noattack' if not_attacking else ''}"))

    df = reduce(lambda left, right: pd.merge(left, right, on=['round'], how='outer'), metrics_df)
    # if model == 'FEMNIST':
    df = df[df["round"] <= 520]
    return df

def build_continuous_static_plot(name, df, model, leftmost=True):
    colors, linestyles = get_colorful_styles()

    if model == 'FEMNIST':
        configs = {
            # 'None_None_None_pgd_noattack': {'label': 'No attack', 'marker': None},
            'l2_2.0_0.08333_pgd': {'label': '2.0', 'marker': get_markers()[0], 'color': colors[1]},
            'l2_4.0_0.16667_pgd': {'label': '4.0', 'marker': get_markers()[1], 'color': colors[2]},
            'l2_10.0_0.416667_pgd': {'label': '10.0', 'marker': get_markers()[2], 'color': colors[4]},
            'None_None_None_pgd': {'label': 'None', 'marker': get_markers()[3], 'color': colors[0]},
        }
        bound_type = '$L_2$-B:'
    elif model == 'FEMNIST_linf':
        configs = {
            'linf_0.001_None_pgd': {'label': '1e-3', 'marker': get_markers()[0], 'color': colors[1]},
            'linf_0.01_None_pgd': {'label': '1e-2', 'marker': get_markers()[1], 'color': colors[2]},
            'linf_0.5_None_pgd': {'label': '5e-1', 'marker': get_markers()[2], 'color': colors[4]},
            'None_None_None_pgd_noattack': {'label': 'No att.', 'marker': get_markers()[4], 'color': colors[0]},
            # 'None_None_None_pgd': {'label': 'None', 'marker': get_markers()[3], 'color': colors[0]},
        }
        bound_type = '$L_\infty$-B:'
    elif model == 'CIFAR10':
        configs = {
            'l2_2.0_0.025_pgd': {'label': '2.0', 'marker': get_markers()[0], 'color': colors[1]},
            'l2_5.0_0.0625_pgd': {'label': '5.0', 'marker': get_markers()[1], 'color': colors[3]},

            # 'l2_10.0_0.125_pgd': {'label': '10.0', 'marker': 's'},
            # 'l2_10.0_0.125_pgd': {'label': '10.0', 'marker': 'v'}
            # 'l2_20.0_0.25_pgd': {'label': '20.0', 'marker': 'o'},
            'l2_30.0_0.375_pgd': {'label': '30.0', 'marker': get_markers()[2], 'color': colors[5]},
            'None_None_None_pgd': {'label': 'None', 'marker': get_markers()[3], 'color': colors[0]},
        }
        bound_type = '$L_2$-B:'

    markevery = 100
    window_size = 20
    markersize = 8

    setup_plt(square=True)

    with PdfPages(f"{output_dir}/{name}.pdf") as pdf:

        fig, ax = plt.subplots()

        # custom_lines_colors_names = ['None', '2.0', '4.0', '5.0', '10.0', '30.0'] # Dont forget to change if we change data
        # custom_lines_colors = [Line2D([0], [0], linestyle="-", lw=2, color=colors[i]) for i in range(len(custom_lines_colors_names))]
        custom_lines_colors = []
        custom_lines_colors_names = []
        plines = []
        for index, suffix in enumerate(configs.keys()):
            values_acc = df[f"accuracy_{suffix}"]
            values_adv = df[f"adv_success_{suffix}"]
            labels = df[f"round"]
            config = configs[suffix]

            plines += plt.plot(labels, values_acc.rolling(window_size).mean().shift(-window_size),
                     linestyle='dotted', label=config['label'], color=colors[index],
                     linewidth=2, marker=config['marker'], markevery=markevery)
            plines += plt.plot(labels, values_adv.rolling(window_size).mean().shift(-window_size),
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

        # Legend
        # if leftmost:
        line = Line2D([0], [0])
        line.set_visible(False)
        custom_lines_colors = [line] + custom_lines_colors
        custom_lines_colors_names = [bound_type] + custom_lines_colors_names

        leg1 = plt.legend(custom_lines_colors, custom_lines_colors_names,
                           bbox_to_anchor=(1.02, 1.), loc=4, ncol=6, columnspacing=0.75)
        ax.add_artist(leg1)

        # if leftmost:
        for vpack in leg1._legend_handle_box.get_children()[:1]:
            for hpack in vpack.get_children():
                del hpack._children[0]

        # if not leftmost:
        #     custom_lines_styles = [Line2D([0], [0], linestyle="-", lw=2, color=COLOR_GRAY),
        #                            Line2D([0], [0], linestyle=":", lw=2, color=COLOR_GRAY)]
        #     leg_task = plt.legend(custom_lines_styles, ["Backdoor", "Main"],
        #                       bbox_to_anchor=(1., 0.05), loc=4, ncol=1,
        #                       columnspacing=0.75)
        #     ax.add_artist(leg_task)

        # plt.legend(title='Bound', mode="expand", loc="lower left", labelspacing=.05, bbox_to_anchor=(1.01, 0, .6, 0))

        pdf.savefig(bbox_inches='tight', pad_inches=0)
        plt.savefig(f"{output_dir}/{name}.png", bbox_inches='tight', pad_inches=0)
        plt.close()

    return fig, df


def build_continuous_median_plot(name, df, model, leftmost=False):

    if model == 'FEMNIST':
        configs = {
            'median_l2_1.0_1.0_pgd': {'label': '1.0', 'marker': '*'},
            'median_l2_5.0_5.0_pgd': {'label': '5.0', 'marker': 'o'},
            # 'median_l2_10.0_10.0': {'label': '10.0', 'marker': 'v'},
            'median_l2_15.0_15.0_pgd': {'label': '15.0', 'marker': 'D'},
            'None_None_None_pgd': {'label': 'None', 'marker': 's'},
        }
    elif model == 'CIFAR10':
        configs = {
            'median_l2_1.0_1.0_pgd': {'label': '1.0', 'marker': '*'},
            # 'median_l2_2.0_2.0_pgd': {'label': '2.0', 'marker': '2'},
            # 'l2_10.0_0.125_pgd': {'label': '10.0', 'marker': 'v'}
            'median_l2_5.0_5.0_pgd': {'label': '5.0', 'marker': 'o'},
            # 'median_l2_10.0_10.0_pgd': {'label': '10.0', 'marker': 'o'},

            'median_l2_15.0_15.0_pgd': {'label': '15.0', 'marker': 'D'},
            'None_None_None_pgd': {'label': 'None', 'marker': 's'},
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
                     linestyle='dotted', label=config['label'], color=colors[index],
                     linewidth=2, marker=config['marker'], markevery=markevery)
            plt.plot(labels, values_adv.rolling(window_size).mean().shift(-window_size),
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

        # if leftmost:
        #     plt.ylabel('Accuracy')
        plt.xlabel('Round')

        # Legend
        # if leftmost:
        line = Line2D([0], [0])
        line.set_visible(False)
        custom_lines_colors = [line] + custom_lines_colors
        custom_lines_colors_names = ['$r$-M:'] + custom_lines_colors_names

        leg1 = plt.legend(custom_lines_colors, custom_lines_colors_names,
                          bbox_to_anchor=(1.02, 1.), loc=4, ncol=6, columnspacing=0.75)
        ax.add_artist(leg1)

        # if leftmost:
        for vpack in leg1._legend_handle_box.get_children()[:1]:
            for hpack in vpack.get_children():
                del hpack._children[0]

        if not leftmost:
            custom_lines_styles = [Line2D([0], [0], linestyle="-", lw=2, color=COLOR_GRAY),
                                   Line2D([0], [0], linestyle=":", lw=2, color=COLOR_GRAY)]
            leg_task = plt.legend(custom_lines_styles, ["Backdoor task", "Main task"],
                                  bbox_to_anchor=(1., 0.07), loc=4, ncol=1,
                                  columnspacing=0.75)
            ax.add_artist(leg_task)


        pdf.savefig(bbox_inches='tight', pad_inches=0)
        plt.savefig(f"{output_dir}/{name}.png", bbox_inches='tight', pad_inches=0)
        plt.close()

    return fig, df


def build_continuous_static_plot_presentation(name, df, model):

    baseline = 'None_None_None_blackbox'
    single_target = None
    title = None
    if model == 'TOO_TIGHT':
        single_target = 'l2_2.0_0.025_pgd'
        title = "Bound too tight ($L_2 \leq 2$)"
    elif model == 'RIGHT':
        single_target = 'l2_5.0_0.0625_pgd'
        title = "Bound ideal ($L_2 \leq 5$)"
    elif model == 'TOO_LOOSE':
        single_target = 'l2_15.0_0.1875_pgd'
        title = "Bound too loose ($L_2 \leq 15$)"

    markevery = 100
    window_size = 20

    setup_plt()

    with PdfPages(f"{output_dir}/{name}.pdf") as pdf:

        fig, ax = plt.subplots()
        colors, linestyles = get_colorful_styles()

        custom_lines_colors = []
        custom_lines_colors_names = []

        index = 0
        suffix = baseline
        values_acc = df[f"accuracy_{suffix}"]
        values_adv = df[f"adv_success_{suffix}"]
        labels = df[f"round"]
        # config = configs[suffix]

        plt.plot(labels, values_acc.rolling(window_size).mean(),
                 linestyle='dotted', label="Main Task (baseline)", color=colors[index],
                 linewidth=2, marker='s', markevery=markevery)

        index = 1
        suffix = single_target
        values_acc = df[f"accuracy_{suffix}"]
        values_adv = df[f"adv_success_{suffix}"]
        labels = df[f"round"]

        plt.plot(labels, values_acc.rolling(window_size).mean(),
                 linestyle='dotted', label="Main Task", color=colors[index],
                 linewidth=2, marker='o', markevery=markevery)
        plt.plot(labels, values_adv.rolling(window_size).mean(),
                 color=colors[index], label="Backdoor Task",
                 linewidth=2, marker='v', markevery=markevery)

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
        plt.title(title)

        # Legend
        plt.legend(bbox_to_anchor=(1., -0.04, 1., .102), loc=3, ncol=1,
                          columnspacing=0.75)

        # plt.legend(title='Bound', mode="expand", loc="lower left", labelspacing=.05, bbox_to_anchor=(1.01, 0, .6, 0))

        pdf.savefig(bbox_inches='tight', pad_inches=0)
        plt.savefig(f"{output_dir}/{name}.png", bbox_inches='tight', pad_inches=0)
        plt.close()

    return fig, df