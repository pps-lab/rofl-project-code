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

def load_continuous_bound_data(model, type):

    if model == 'FEMNIST':
        query = {
            # "meta.description": "FEMNIST continuous under different bounds",
            "$or": [
                {"meta.description": "FEMNIST continuous under different bounds"},
                {"meta.description": "Test adding neurotoxin"},
                # {"meta.description": "FEMNIST continuous anticipate"},
                {"meta.description": "FEMNIST continuous anticipate both"}
            ]
            # 'hyperparameters.args.client.malicious.attack_stop': { '$ne': 0 }
        }
    elif model == 'CIFAR10':
        # rgx = re.compile('.*cifar_continuous_bound_racingstripes_1618325970_.*', re.IGNORECASE)  # compile the regex
        #
        # query = {
        #     "meta.description": "CIFAR10 Continuous under different static bounds",
        #     "_id": rgx
        # }
        # desc = "CIFAR10 static bounds for green cars"
        if type == "median":
            query = {
                "$or": [
                    { "meta.description": "CIFAR10 green car median" },
                    { "meta.description": "CIFAR10 continuous anticipate" },
                    { "meta.description": "CIFAR10 continuous neurotoxin" }
                ]
            }
        else:
            query = {
                "$or": [
                    { "meta.description": "CIFAR10 static bounds for green cars"},
                    { "meta.description": "CIFAR10 continuous anticipate" },
                    { "meta.description": "CIFAR10 continuous neurotoxin" }
                ]
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
        attack_method = get_attack_method(metric)
        print("Combination: ", f"{clip_type}_{bound}_{attack_method}{'_noattack' if not_attacking else ''}")
        metrics_df.append(_preprocess(df, f"{clip_type}_{bound}_{attack_method}{'_noattack' if not_attacking else ''}"))

    df = reduce(lambda left, right: pd.merge(left, right, on=['round'], how='outer'), metrics_df)
    # if model == 'FEMNIST':
    df = df[df["round"] <= 520]
    return df

def get_attack_method(metric):
    if metric['client']['malicious']['evasion'] == None:
        return "blackbox"
    elif metric['client']['malicious']['objective']['name'] == 'TargetedAttack':
        if metric['client']['malicious']['evasion']['name'] == 'NormBoundPGDEvasion':
            if metric['client']['malicious']['evasion']['args']['pgd_factor'] is not None:
                return "pgd"
            else:
                return "mr"
        elif metric['client']['malicious']['evasion']['name'] == 'NeurotoxinEvasion':
            return "neurotoxin"
        else:
            return "ERROR"
    elif metric['client']['malicious']['objective']['name'] == 'AnticipateTfAttack':
        return "anticipate"
    return "ERROR2"

def build_plot(name, df, model, leftmost, configs, type):

    colors = get_colors_attackers()

    markevery = 100
    window_size = 20

    setup_plt(square=True)

    # plt_new_attacks = model == 'FEMNIST' # for now

    with PdfPages(f"{output_dir}/{name}.pdf") as pdf:

        fig, ax = plt.subplots()

        custom_lines_colors = []
        custom_lines_colors_names = []
        for index, suffix in enumerate(configs.keys()):

            if suffix == 'None_None_mr':
                values_pgd = df[f"adv_success_{suffix}"]
            else:
                # if suffix == 'l2_1.0':
                #     values_pgd = df[f"adv_success_l2_2.0_pgd"]
                # else:
                values_pgd = df[f"adv_success_{suffix}_pgd"]
            # values_blackbox = df[f"adv_success_{suffix}_blackbox"]
            values_anticipate = df[f"adv_success_{suffix}_anticipate"]
            labels = df[f"round"]
            config = configs[suffix]

            plt.plot(labels, values_pgd.rolling(window_size).mean().shift(-window_size),
                     linestyle=LINESTYLE_PGD, label=config['label'], color=config['color'],
                     linewidth=2, marker=config['marker'], markevery=markevery)

            plt.plot(labels, values_anticipate.rolling(window_size).mean().shift(-window_size),
                     linestyle=LINESTYLE_AT, label=config['label'], color=config['color'],
                     linewidth=2, marker=config['marker'], markevery=markevery)
            if f"adv_success_{suffix}_neurotoxin" in df.columns:
                values_neurotoxin = df[f"adv_success_{suffix}_neurotoxin"]
                plt.plot(labels, values_neurotoxin.rolling(window_size).mean().shift(-window_size),
                         linestyle=LINESTYLE_NT, label=config['label'], color=config['color'],
                         linewidth=2, marker=config['marker'], markevery=markevery)
            else:
                print("NEUROTOXIN MISSING")

            # plt.plot(labels, values_blackbox.rolling(window_size).mean().shift(-window_size),
            #          linestyle=LINESTYLE_DP,
            #          color=config['color'],
            #          linewidth=2, marker=config['marker'], markevery=markevery, path_effects=[DP_PATHEFFECT])

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
        type_bound = '$L_2$-B:' if type == 'l2' else 'M-$r$:'
        custom_lines_colors_names = [type_bound] + custom_lines_colors_names

        leg1 = plt.legend(custom_lines_colors,
                          custom_lines_colors_names,
                          bbox_to_anchor=(1.02, 1.), loc=4, ncol=6, columnspacing=0.75)

        for vpack in leg1._legend_handle_box.get_children()[:1]:
            for hpack in vpack.get_children():
                del hpack._children[0]
        ax.add_artist(leg1)

        # if not leftmost:
        custom_lines_styles = [Line2D([0], [0], linestyle=LINESTYLE_AT, lw=2, color=COLOR_GRAY),
                               Line2D([0], [0], linestyle=LINESTYLE_PGD, lw=2, color=COLOR_GRAY),
                               Line2D([0], [0], linestyle=LINESTYLE_NT, lw=2, color=COLOR_GRAY),
                               # Line2D([0], [0], linestyle=LINESTYLE_DP, lw=2, color=COLOR_GRAY,
                               #        path_effects=[DP_PATHEFFECT])
                               ]
        leg_task = plt.legend(custom_lines_styles, [LABEL_AT, LABEL_PGD, LABEL_NT],
                              bbox_to_anchor=(0.0, 1.0), loc="upper left", ncol=3,
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
