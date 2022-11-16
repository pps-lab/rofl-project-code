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

def load_data(model):

    if model == 'FEMNIST':
        query = {
            # "meta.description": "FEMNIST continuous under different bounds",
            "$or": [
                {"meta.description": "FEMNIST untargeted"}
            ]
            # 'hyperparameters.args.client.malicious.attack_stop': { '$ne': 0 }
        }
    elif model == 'CIFAR10':
        query = {
            "meta.description": "CIFAR untargeted",
        }
    elif model == 'BOTH':
        query = {
            "$or": [
                {"meta.description": "FEMNIST untargeted"},
                {"meta.description": "CIFAR untargeted"}
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
        not_attacking = determine_attack(metric)
        task = 'CIFAR10' if metric['dataset']['dataset'] == 'cifar10' else 'FEMNIST'
        clip_type = metric['client']['clip']['type'] \
                            if metric['client']['clip'] is not None else None
        bound = metric['hyperparameters']['args']['client']['clip']['value'] \
                            if metric['hyperparameters']['args']['client']['clip'] is not None else None

        print(f"Combination: ", f"{task}_{clip_type}_{bound}{'_noattack' if not_attacking else ''}")
        metrics_df.append(_preprocess(df, f"{task}_{clip_type}_{bound}{'_noattack' if not_attacking else ''}"))

    df = reduce(lambda left, right: pd.merge(left, right, on=['round'], how='outer'), metrics_df)

    df = df[df["round"] <= 520]
    return df

def determine_attack(metric):
    if 'attack_stop' in metric['client']['malicious'] \
        and metric['client']['malicious']['attack_stop'] < 1000:
        return 'noattack'
    else:
        if "attack_start" in metric['client']['malicious'] \
            and metric['client']['malicious']['attack_start'] > 10:
            return 'noattack'
        else:
            return None

def build_plot(name, df, model, bound_type, leftmost=True, grid=True, full_legend=False):
    colors, linestyles = get_colorful_styles()

    if model == 'FEMNIST':
        if bound_type == "l2":
            configs = {
                f'{bound_type}_2.0': {'label': '2.0', 'marker': get_markers()[0], 'color': colors[1]},
                f'{bound_type}_4.0': {'label': '4.0', 'marker': get_markers()[1], 'color': colors[2]},
                f'{bound_type}_10.0': {'label': '10.0', 'marker': get_markers()[2], 'color': colors[4]},

                f'{bound_type}_20.0': {'label': '20.0', 'marker': get_markers()[2], 'color': colors[4]},
                f'{bound_type}_30.0': {'label': '30.0', 'marker': get_markers()[2], 'color': colors[4]},
                f'{bound_type}_40.0': {'label': '40.0', 'marker': get_markers()[2], 'color': colors[4]},
                f'{bound_type}_50.0': {'label': '50.0', 'marker': get_markers()[2], 'color': colors[4]},

                f'None_None': {'label': 'None', 'marker': get_markers()[0], 'color': colors[1]}
            }
            bound_type = '$L_2$-B:' if not full_legend else '$L_2$-Bound:'
        elif bound_type == "median_l2":
            configs = {
                f'{bound_type}_1.0': {'label': '1.0', 'marker': get_markers()[0], 'color': colors[1]},
                f'{bound_type}_5.0': {'label': '5.0', 'marker': get_markers()[1], 'color': colors[2]},
                f'{bound_type}_15.0': {'label': '15.0', 'marker': get_markers()[2], 'color': colors[4]},

                f'{bound_type}_30.0': {'label': '30.0', 'marker': get_markers()[2], 'color': colors[4]},
                f'{bound_type}_40.0': {'label': '40.0', 'marker': get_markers()[2], 'color': colors[4]},
                f'{bound_type}_50.0': {'label': '50.0', 'marker': get_markers()[2], 'color': colors[4]},

                f'None_None': {'label': 'None', 'marker': get_markers()[0], 'color': colors[1]}
            }
        elif bound_type == "linf":
            bound_vals = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0]
            configs = { f'{bound_type}_{bnd}': {'label': f'{bnd}', 'marker': get_markers()[0], 'color': colors[1]} for bnd in bound_vals }
            configs[f'None_None'] = {'label': 'None', 'marker': get_markers()[0], 'color': colors[1]}

            bound_type = '$L_\infty$-B:' if not full_legend else '$L_\infty$-Bound:'
    elif model == 'CIFAR10':
        if bound_type == "l2":
            configs = {
                f'{bound_type}_1.0': {'label': '2.0', 'marker': get_markers()[0], 'color': colors[1]},
                f'{bound_type}_2.0': {'label': '2.0', 'marker': get_markers()[1], 'color': colors[2]},
                f'{bound_type}_5.0': {'label': '5.0', 'marker': get_markers()[1], 'color': colors[2]},
                f'{bound_type}_10.0': {'label': '10.0', 'marker': get_markers()[1], 'color': colors[2]},
                f'{bound_type}_20.0': {'label': '20.0', 'marker': get_markers()[1], 'color': colors[2]},

                f'{bound_type}_50.0': {'label': '50.0', 'marker': get_markers()[1], 'color': colors[2]},
                f'{bound_type}_100.0': {'label': '100.0', 'marker': get_markers()[1], 'color': colors[2]},
                f'{bound_type}_150.0': {'label': '150.0', 'marker': get_markers()[1], 'color': colors[2]},
                f'{bound_type}_200.0': {'label': '200.0', 'marker': get_markers()[1], 'color': colors[2]},

                f'{bound_type}_30.0': {'label': '30.0', 'marker': get_markers()[2], 'color': colors[4]},
                f'None_None': {'label': 'None', 'marker': get_markers()[0], 'color': colors[1]}
            }
            bound_type = '$L_2$-B:' if not full_legend else '$L_2$-Bound:'
        elif bound_type == "median_l2":
            configs = {
                f'{bound_type}_1.0': {'label': '1.0', 'marker': get_markers()[0], 'color': colors[1]},
                f'{bound_type}_5.0': {'label': '5.0', 'marker': get_markers()[1], 'color': colors[2]},
                f'{bound_type}_10.0': {'label': '5.0', 'marker': get_markers()[1], 'color': colors[2]},
                f'{bound_type}_15.0': {'label': '15.0', 'marker': get_markers()[2], 'color': colors[4]},

                f'{bound_type}_30.0': {'label': '30.0', 'marker': get_markers()[2], 'color': colors[4]},
                f'{bound_type}_50.0': {'label': '50.0', 'marker': get_markers()[2], 'color': colors[4]},
                f'{bound_type}_75.0': {'label': '75.0', 'marker': get_markers()[2], 'color': colors[4]},
                f'{bound_type}_100.0': {'label': '100.0', 'marker': get_markers()[2], 'color': colors[4]},

                f'None_None': {'label': 'None', 'marker': get_markers()[0], 'color': colors[1]}
            }
        elif bound_type == "linf":
            configs = {
                f'{bound_type}_0.01': {'label': '0.01', 'marker': get_markers()[0], 'color': colors[1]},
                f'{bound_type}_0.05': {'label': '0.05', 'marker': get_markers()[1], 'color': colors[2]},
                f'{bound_type}_5.0': {'label': '5.0', 'marker': get_markers()[1], 'color': colors[2]},
                f'{bound_type}_10.0': {'label': '10.0', 'marker': get_markers()[2], 'color': colors[4]},
                f'None_None': {'label': 'None', 'marker': get_markers()[0], 'color': colors[1]}
            }
            bound_type = '$L_\infty$-B:' if not full_legend else '$L_\infty$-Bound:'


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
        # if leftmost:
        line = Line2D([0], [0])
        line.set_visible(False)
        custom_lines_colors = [line] + custom_lines_colors
        custom_lines_colors_names = [bound_type] + custom_lines_colors_names

        leg1 = plt.legend(custom_lines_colors, custom_lines_colors_names,
                           bbox_to_anchor=(1.02, 1.), loc=4, ncol=6, columnspacing=0.75)
        ax.add_artist(leg1)

        # if leftmost:

        # custom organization
        # for vpack in leg1._legend_handle_box.get_children()[:1]:
        #     for hpack in vpack.get_children():
        #         del hpack._children[0]

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


def build_plot_attack_impact(name, df, model, bound_type, bounds, leftmost=True, grid=True, full_legend=False):
    colors, linestyles = get_colorful_styles()

    unattacked_key = f"accuracy_{model}_None_None_noattack"

    markevery = 100
    window_size = 20
    markersize = 8

    setup_plt(square=True)

    def get_val(x):
        # return x[x["round"] == 500]
        return x[(x["round"] > 480) & (x["round"] <= 500)].mean(axis=0)

    max_accuracy = get_val(df)[unattacked_key].item()

    with PdfPages(f"{output_dir}/{name}.pdf") as pdf:

        fig, ax = plt.subplots()

        custom_lines_colors = []
        custom_lines_colors_names = []
        plines = []

        # get top acc
        x = get_val(df)
        y = pd.DataFrame(columns=['bound', 'accuracy'])
        for b in bounds:
            y = y.append({ 'bound': b, 'accuracy': float(x[f"accuracy_{model}_{bound_type}_{b}"]) }, ignore_index=True)

        # y = y.append({ 'bound': 25.0, 'accuracy': float(x[f"accuracy_None_None"]) }, ignore_index=True)


        plt.plot(y['bound'], max_accuracy - y['accuracy'], '-o', linewidth=2, color=colors[1] )
        plt.axhline(max_accuracy - float(x[f"accuracy_None_None"]), linestyle='dashed')

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
        # ax.get_yaxis().set_major_formatter(
        #     matplotlib.ticker.FuncFormatter(yaxis_formatter))
        if bound_type == "linf":
            ax.set_xscale('log')

        if leftmost:
            plt.ylabel('Attack Impact')
        plt.xlabel(f'{bound_type}')

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

        # custom organization
        # for vpack in leg1._legend_handle_box.get_children()[:1]:
        #     for hpack in vpack.get_children():
        #         del hpack._children[0]

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


def build_plot_attack_impact_combined(name, df, model, bounds_map: dict, yticks, leftmost=True, grid=True, full_legend=False):
    colors, linestyles = get_colorful_styles()

    unattacked_key = f"accuracy_{model}_None_None_noattack"

    markevery = 100
    window_size = 20
    markersize = 8

    plt_params = {
        'backend': 'ps',
        'axes.labelsize': 20,
        'legend.fontsize': 16,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'font.size': 18,
        'figure.figsize': [10, 10. / 3.],
        'font.family': 'Times New Roman',
        'font.weight': 'normal'
    }
    plt.rcParams.update(plt_params)
    plt.rc('pdf', fonttype=42)  # IMPORTANT to get rid of Type 3

    def get_val(x):
        # return x[x["round"] == 500]
        return x[(x["round"] > 480) & (x["round"] <= 500)].mean(axis=0)

    max_accuracy = get_val(df)[unattacked_key].item()

    bound_labels = {
        'l2': '$L_2$',
        'median_l2': 'M-$r$',
        'linf': '$L_\\infty$'
    }

    with PdfPages(f"{output_dir}/{name}.pdf") as pdf:

        fig, ax = plt.subplots(1, 3)

        custom_lines_colors = []
        custom_lines_colors_names = []
        plines = []

        # get top acc
        x = get_val(df)
        for idx, (bound_type, bounds) in enumerate(bounds_map.items()):
            y = pd.DataFrame(columns=['bound', 'accuracy'])
            for b in bounds:
                y = y.append({ 'bound': b, 'accuracy': float(x[f"accuracy_{model}_{bound_type}_{b}"]) }, ignore_index=True)

            ax[idx].plot(y['bound'], max_accuracy - y['accuracy'], '-o', linewidth=2, color=colors[1] )
            # ax[idx].axhline(max_accuracy - float(x[f"accuracy_None_None"]), linestyle='dashed')

            if bound_type == "linf":
                ax[idx].set_xscale('log')

            ax[idx].set_box_aspect(1)
            ax[idx].set_xlabel(f'{bound_labels[bound_type]}')
            # ax[idx].set_yticks([])
            ax[idx].grid(True, axis="y", linestyle=':', color='0.6', zorder=0, linewidth=1.2)
            ax[idx].set_yticks([0, 0.25, 0.5, 0.75, 1])

            ax[idx].set_ylim(ymin=-0.05, ymax=1.05)

        ax[0].set_yticks(yticks)
        ax[1].yaxis.set_ticklabels([])
        ax[2].yaxis.set_ticklabels([])
        ax[0].set_ylabel('Attack Impact')

        plt.title(model)

    ##########################
        # General Format
        ##########################
        ##########################
        # Y - Axis Format
        ##########################
        # plt.ylim(ymin=-0.05, ymax=1.05)
        # plt.yticks([0, 0.25, 0.5, 0.75, 1])
        # ax.get_yaxis().set_major_formatter(
        #     matplotlib.ticker.FuncFormatter(yaxis_formatter))


        # Legend
        # if leftmost:
        # if leftmost:

        # custom organization
        # for vpack in leg1._legend_handle_box.get_children()[:1]:
        #     for hpack in vpack.get_children():
        #         del hpack._children[0]

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

def build_plot_attack_impact_combined_singleplot(name, df, bounds_map: dict, yticks, leftmost=True, grid=True, full_legend=False):
    colors, linestyles = get_colorful_styles()


    markevery = 100
    window_size = 20
    markersize = 8

    plt_params = {
        'backend': 'ps',
        'axes.labelsize': 20,
        'legend.fontsize': 16,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'font.size': 14,
        'figure.figsize': [10, 22. / 3.],
        'font.family': 'Times New Roman',
        'font.weight': 'normal'
    }
    plt.rcParams.update(plt_params)
    plt.rc('pdf', fonttype=42)  # IMPORTANT to get rid of Type 3

    def get_val(x):
        # return x[x["round"] == 500]
        return x[(x["round"] > 480) & (x["round"] <= 500)].mean(axis=0)

    bound_labels = {
        'l2': '$L_2$',
        'median_l2': 'M-$r$',
        'linf': '$L_\\infty$'
    }

    with PdfPages(f"{output_dir}/{name}.pdf") as pdf:

        fig, ax = plt.subplots(2, 3)

        custom_lines_colors = []
        custom_lines_colors_names = []
        plines = []

        # get top acc
        for row_idx, task in enumerate(['FEMNIST', 'CIFAR10']):
            unattacked_key = f"accuracy_{task}_None_None_noattack"
            max_accuracy = get_val(df)[unattacked_key].item()

            x = get_val(df)
            for idx, (bound_type, bounds) in enumerate(bounds_map[task].items()):
                y = pd.DataFrame(columns=['bound', 'accuracy'])
                for b in bounds:
                    y = y.append({ 'bound': b, 'accuracy': float(x[f"accuracy_{task}_{bound_type}_{b}"]) }, ignore_index=True)

                # print(max_accuracy, y)

                ax[row_idx, idx].plot(y['bound'], max_accuracy - y['accuracy'], '-o', linewidth=2, color=colors[1] )
                # ax[idx].axhline(max_accuracy - float(x[f"accuracy_None_None"]), linestyle='dashed')

                if bound_type == "linf":
                    ax[row_idx, idx].set_xscale('log')

                ax[row_idx, idx].set_box_aspect(1)
                # ax[row_idx, idx].set_xlabel(f'{bound_labels[bound_type]}')
                # ax[idx].set_yticks([])
                ax[row_idx, idx].grid(True, axis="y", linestyle=':', color='0.6', zorder=0, linewidth=1.2)
                ax[row_idx, idx].set_yticks([0, 0.25, 0.5, 0.75, 1])

                ax[row_idx, idx].set_ylim(ymin=-0.05, ymax=1.05)

                ax[row_idx, idx].set_title(f"{task} + {bound_labels[bound_type]}")

            ax[row_idx, 0].set_yticks(yticks)
            ax[row_idx, 1].yaxis.set_ticklabels([])
            ax[row_idx, 2].yaxis.set_ticklabels([])
            ax[row_idx, 0].set_ylabel('Attack Impact ($\\%$)')

            ax[row_idx, 0].get_yaxis().set_major_formatter(
                matplotlib.ticker.FuncFormatter(yaxis_formatter))

            ax[row_idx, 1].set_xticks([0, 10, 20, 30, 40, 50]) # median

        ax[row_idx, 2].set_xticks([0.01, 0.1, 10]) # median

        # plt.title(task)

        ##########################
        # General Format
        ##########################
        ##########################
        # Y - Axis Format
        ##########################
        # plt.ylim(ymin=-0.05, ymax=1.05)
        # plt.yticks([0, 0.25, 0.5, 0.75, 1])


        # Legend
        # if leftmost:
        # if leftmost:

        # custom organization
        # for vpack in leg1._legend_handle_box.get_children()[:1]:
        #     for hpack in vpack.get_children():
        #         del hpack._children[0]

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