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
from matplotlib.patches import Rectangle

from common import _preprocess, setup_plt, query_data, get_colorful_styles, output_dir, COLOR_GRAY, yaxis_formatter, get_markers


def load_params_data():
    weights = pd.read_csv("./data/weight_distribution/params_outside_bound_0.01.csv", index_col=0)

    total_weights = 44426.0

    # Convert to real array
    weights['clients'] = weights['clients'].apply(lambda x: np.array(x.strip('][').split(', '), dtype=np.int32))

    weights['honest_stddev'] = weights['clients'].apply(lambda x: np.std(x[:-1]))
    weights['honest_mean'] = weights['clients'].apply(lambda x: np.mean(x[:-1]))
    weights['mal'] = weights['clients'].apply(lambda x: x[-1])

    weights['honest_stddev_perc'] = weights['honest_stddev'] / total_weights
    weights['honest_mean_perc'] = weights['honest_mean'] / total_weights
    weights['mal_perc'] = weights['mal'] / total_weights

    weights = weights[weights["round"] <= 520]

    return weights


def build_continuous_static_plot(name, df):
    colors, linestyles = get_colorful_styles()

    configs = {
        'honest_mean_perc': {'label': 'Benign', 'marker': get_markers()[0], 'color': colors[1] },
        'mal_perc': {'label': 'Malicious', 'marker': get_markers()[1], 'color': colors[0] },
    }

    markevery = 100
    window_size = 20
    markersize = 8
    error_color = "0.85"

    setup_plt(square=True)

    with PdfPages(f"{output_dir}/{name}.pdf") as pdf:

        fig, ax = plt.subplots()

        custom_lines_colors = []
        custom_lines_colors_names = []

        ##################
        # Our 0.5% bound #
        ##################
        plt.axhline(y=0.005, color=COLOR_GRAY, linestyle='dashed')

        for index, suffix in enumerate(configs.keys()):
            values_acc = df[suffix]
            labels = df[f"round"]
            config = configs[suffix]

            if suffix == "mal_perc":
                plt.plot(labels, values_acc.rolling(window_size).mean().shift(-window_size),
                         linestyle='solid', label=config['label'], color=config['color'],
                         linewidth=2, marker=config['marker'], markevery=markevery)

                values_std = values_acc.rolling(window_size).std().shift(-window_size)
                plt.fill_between(labels,
                                 values_acc - values_std,
                                 values_acc + values_std,
                                 alpha=1, edgecolor='#3F7F4C', facecolor=error_color, linewidth=0)
            else:
                #values_acc.rolling(window_size).mean().shift(-window_size)
                plt.plot(labels, values_acc.rolling(window_size).mean().shift(-window_size), #df['honest_stddev_perc'].rolling(window_size).mean().shift(-window_size),
                         linestyle='solid', label=config['label'], color=config['color'],
                         linewidth=2, marker=config['marker'], markevery=markevery)

                values_std = df['honest_stddev_perc'].rolling(window_size).mean().shift(-window_size)
                plt.fill_between(labels,
                                values_acc - values_std,
                                values_acc + values_std,
                                alpha=1, edgecolor='#3F7F4C', facecolor=error_color, linewidth=0)

            custom_lines_colors.append(Line2D([0], [0], linestyle="-", lw=2, marker=config['marker'], color=config['color']))
            custom_lines_colors_names.append(config['label'])

        ##########################
        # General Format
        ##########################
        ax.grid(True, axis="y", linestyle=':', color='0.6', zorder=0, linewidth=1.2)
        ##########################
        # Y - Axis Format
        ##########################
        # ax.set_ylim(ymin=-0.05, ymax=0.455)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.get_yaxis().set_major_formatter(
            matplotlib.ticker.FuncFormatter(yaxis_formatter))

        plt.ylabel('% Weights above bound')
        plt.xlabel('Round')

        # Legend
        # if leftmost:
        line = Line2D([0], [0])
        line.set_visible(False)
        custom_lines_colors = [line] + custom_lines_colors
        custom_lines_colors_names = ['Update type:'] + custom_lines_colors_names

        leg1 = plt.legend(custom_lines_colors, custom_lines_colors_names,
                           bbox_to_anchor=(1.02, 1.), loc=4, ncol=6, columnspacing=0.75)
        ax.add_artist(leg1)


        indicator_lines =[
            Line2D([0], [0], linestyle="dashed", lw=2, color=COLOR_GRAY)]
        indicator_lines_labels = ["$p_v=0.005$"]
        leg_line = plt.legend(indicator_lines, indicator_lines_labels,
                          bbox_to_anchor=(1., 1.), loc='upper right', ncol=1,
                        columnspacing=0.75)
        ax.add_artist(leg_line)

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


def load_weight_distribution_single_round_data():
    dir = "./data/weight_distribution/updates"
    mal = np.load(os.path.join(dir, "2245_m_1.npy"), allow_pickle=True)
    ben = np.load(os.path.join(dir, "32_b_1.npy"), allow_pickle=True)

    df = pd.DataFrame()

    for label, update in zip(["mal", "ben"], [mal, ben]):
        flattened = np.concatenate([np.reshape(u, -1) for u in update])
        df_u = pd.DataFrame()
        df_u[f'{label}_weights'] = flattened

        # df_u[(df_u[f'{label}_weights'] > 1) | df_u[f'{label}_weights'] < -1][f'{label}_weights'] = np.NaN

        df = df.merge(df_u, how='right', left_index=True, right_index=True)

    return df


def build_single_round(name, df):

    setup_plt(square=True)

    bound = 0.01

    with PdfPages(f"{output_dir}/{name}.pdf") as pdf:

        fig, ax = plt.subplots()
        colors, linestyles = get_colorful_styles()

        bins = np.linspace(-0.4, 0.4, 40)

        plt.hist(df["mal_weights"], color=colors[0], bins=bins, density=False)
        plt.hist(df["ben_weights"], color=colors[1], bins=bins, alpha=0.5, density=False)

        custom_lines_colors = [
            Rectangle((0,0), 1, 1, facecolor=colors[1]),
            Rectangle((0,0), 1, 1, facecolor=colors[0])
            ]
        custom_lines_colors_names = [
            "Benign",
            "Malicious"
        ]

        plt.axvline(-bound, color=COLOR_GRAY, linestyle='dashed')
        plt.axvline(bound, color=COLOR_GRAY, linestyle='dashed')

        ##########################
        # General Format
        ##########################
        ax.grid(True, axis="y", linestyle=':', color='0.6', zorder=0, linewidth=1.2)
        ##########################
        # Y - Axis Format
        ##########################
        # ax.set_ylim(ymin=-0.05, ymax=1.05)
        # ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        # ax.set_xlim(xmin=-1, xmax=1)

        # if leftmost:
        plt.ylabel('Count')
        plt.xlabel('Weight value')

        indicator_lines = [
            Line2D([0], [0], linestyle="dashed", lw=2, color=COLOR_GRAY)]
        indicator_lines_labels = ["$L_\infty$-B$=0.01$"]
        leg_line = plt.legend(indicator_lines, indicator_lines_labels,
                              bbox_to_anchor=(1., 1.), loc='upper right', ncol=1,
                              columnspacing=0.75)
        ax.add_artist(leg_line)

        # Legend
        # if leftmost:
        line = Line2D([0], [0])
        line.set_visible(False)
        custom_lines_colors = [line] + custom_lines_colors
        custom_lines_colors_names = ['Update type:'] + custom_lines_colors_names

        leg1 = plt.legend(custom_lines_colors, custom_lines_colors_names,
                          bbox_to_anchor=(1.02, 1.), loc=4, ncol=6, columnspacing=0.75)
        ax.add_artist(leg1)

        # if leftmost:
        for vpack in leg1._legend_handle_box.get_children()[:1]:
            for hpack in vpack.get_children():
                del hpack._children[0]

        pdf.savefig(bbox_inches='tight', pad_inches=0)
        plt.savefig(f"{output_dir}/{name}.png", bbox_inches='tight', pad_inches=0)
        plt.close()

    return fig, df


def build_single_round_broken(name, df):
    setup_plt(square=True)

    bound = 0.01

    with PdfPages(f"{output_dir}/{name}.pdf") as pdf:

        fig, (ax, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [1, 2]})
        colors, linestyles = get_colorful_styles()

        bins = np.linspace(-0.4, 0.4, 40)

        ax.hist(df["mal_weights"], color=colors[0], bins=bins, density=False)
        ax.hist(df["ben_weights"], color=colors[1], bins=bins, alpha=0.5, density=False)
        ax2.hist(df["mal_weights"], color=colors[0], bins=bins, density=False)
        ax2.hist(df["ben_weights"], color=colors[1], bins=bins, alpha=0.5, density=False)

        custom_lines_colors = [
            Rectangle((0, 0), 1, 1, facecolor=colors[1]),
            Rectangle((0, 0), 1, 1, facecolor=colors[0])
        ]
        custom_lines_colors_names = [
            "Benign",
            "Malicious"
        ]

        ax.axvline(-bound, color=COLOR_GRAY, linestyle='dashed')
        ax.axvline(bound, color=COLOR_GRAY, linestyle='dashed')
        ax2.axvline(-bound, color=COLOR_GRAY, linestyle='dashed')
        ax2.axvline(bound, color=COLOR_GRAY, linestyle='dashed')

        ##########################
        # General Format
        ##########################
        ax.grid(True, axis="y", linestyle=':', color='0.6', zorder=0, linewidth=1.2)
        ax2.grid(True, axis="y", linestyle=':', color='0.6', zorder=0, linewidth=1.2)
        ##########################
        # Y - Axis Format
        ##########################
        # ax.set_ylim(ymin=-0.05, ymax=1.05)
        # ax.set_yticks([40000])
        # ax.set_xlim(xmin=-1, xmax=1)
        ax.set_yticks(list(range(0, 50000, 2500)))
        ax2.set_yticks(list(range(0, 50000, 2500)))
        ax.set_ylim(ymin=40000-100, ymax=45000)
        ax2.set_ylim(ymin=0, ymax=10000)

        ax.spines['bottom'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        # ax.xaxis.tick_top()
        ax.xaxis.set_ticks_position('none')
        ax.tick_params(labeltop=False)  # don't put tick labels at the top
        ax2.xaxis.tick_bottom()

        d = .015  # how big to make the diagonal lines in axes coordinates
        # arguments to pass to plot, just so we don't keep repeating them
        kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
        ax.plot((-d, +d), (-2 * d, + 2 * d), **kwargs)  # top-left diagonal
        ax.plot((1 - d, 1 + d), (-2 * d, +2 * d), **kwargs)  # top-right diagonal

        kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
        ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
        ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

        ##########################
        # Y - LEGEND
        ##########################
        # if leftmost:
        # plt.ylabel('Count')
        plt.xlabel('Weight value')

        plt.sca(ax)
        indicator_lines = [
            Line2D([0], [0], linestyle="dashed", lw=2, color=COLOR_GRAY)]
        indicator_lines_labels = ["$L_\infty$-B$=0.01$"]
        leg_line = plt.legend(indicator_lines, indicator_lines_labels,
                              bbox_to_anchor=(1., 1.), loc='upper right', ncol=1,
                              columnspacing=0.75)
        ax.add_artist(leg_line)

        # Legend
        # if leftmost:
        line = Line2D([0], [0])
        line.set_visible(False)
        custom_lines_colors = [line] + custom_lines_colors
        custom_lines_colors_names = ['Update type:'] + custom_lines_colors_names

        leg1 = plt.legend(custom_lines_colors, custom_lines_colors_names,
                          bbox_to_anchor=(1.02, 1.), loc=4, ncol=6, columnspacing=0.75)
        ax.add_artist(leg1)

        # if leftmost:
        for vpack in leg1._legend_handle_box.get_children()[:1]:
            for hpack in vpack.get_children():
                del hpack._children[0]

        ### Common y label
        fig.add_subplot(111, frameon=False)
        # hide tick and tick label of the big axes
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        plt.grid(False)
        plt.ylabel("Count", labelpad=20)

        pdf.savefig(bbox_inches='tight', pad_inches=0)
        plt.savefig(f"{output_dir}/{name}.png", bbox_inches='tight', pad_inches=0)
        plt.close()

    return fig, df