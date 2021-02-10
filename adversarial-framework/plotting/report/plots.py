#!/usr/bin/python
# coding=utf-8
import math
import sys
import os

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import scipy
from matplotlib import ticker
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
from matplotlib.font_manager import FontProperties
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
import seaborn as sns
from matplotlib.patches import Patch

from plotting.report.extract_histogram import extract_histogram

plot_data_save_path = "./data/"
plots = "./images/"

COLOR_GRAY = "#AAAAAA"

FONT_SIZE = 20

DATA_KEYS_FMNIST = {
    "CLIP_DEFENSE": {
        "L2": {
            "BASELINE": 'e41_google_tasks_noconstrain_evaluation',
            "XMAX": 50,
            "XMIN": 0.01,
            "ATTACK": {
                'e41_clipl2_0_01_evaluation': 0.01,
                'e41_clipl2_0_025_evaluation': 0.025,
                'e41_clipl2_0_05_evaluation': 0.05,
                'e41_clipl2_0_1_evaluation': 0.1,
                'e41_clipl2_0_5_evaluation': 0.5,
                'e41_clipl2_1_evaluation': 1,
                'e41_clipl2_3_evaluation': 3,
                'e41_clipl2_3_5_evaluation': 3.5,
                'e41_clipl2_5_evaluation': 5,
                'e41_clipl2_10_evaluation': 10,
                'e41_clipl2_12_evaluation': 12,
                'e41_clipl2_14_evaluation': 14,
                'e41_clipl2_16_evaluation': 16,
                'e41_clipl2_18_evaluation': 18,
                'e41_clipl2_20_evaluation': 20,
                'e41_clipl2_25_evaluation': 25,
                'e41_clipl2_30_evaluation': 30,
                'e41_clipl2_35_evaluation': 35,
            },
            "PGD_ATTACK": {
                'e41_clipl2_20_pgd_evaluation': 20,
                'e41_clipl2_10_pgd_evaluation': 10
            },
            "NO_ATTACK": {
                "e41_clipl2_0_01_noattack_evaluation": 0.01,
                "e41_clipl2_0_025_noattack_evaluation": 0.025,
                # "e41_clipl2_0_05_noattack_evaluation": 0.05,
                "e41_clipl2_0_1_noattack_evaluation": 0.1,
                "e41_clipl2_3_5_noattack_evaluation": 3.5,
                "e41_clipl2_35_noattack_evaluation": 35
            }

            # 'e41_clipl2_100_evaluation': 100
        },
        "LINF": {
            "BASELINE": 'e41_google_tasks_noconstrain_evaluation',
            "XMAX": 0.2,
            "XMIN": 0.00004,
            "ATTACK": {
                'e41_clipinf_0_00005_2_evaluation': 0.00005,
                'e41_clipinf_0_0001_evaluation': 0.0001,
                'e41_clipinf_0.00015_evaluation': 0.00015,
                'e41_clipinf_0_00100_evaluation': 0.0010,
                'e41_clipinf_0.0015_evaluation': 0.0015,
                'e41_clipinf_0.005_evaluation': 0.005,
                'e41_clipinf_0.015_evaluation': 0.015,
                'e41_clipinf_0.010_evaluation': 0.01,
                'e41_clipinf_0.020_evaluation': 0.02,
                'e41_clipinf_0.025_evaluation': 0.025,
                'e41_clipinf_0_03_evaluation': 0.03,
                # 'e41_emnist_clipinf_0_03_evaluation': 0.03,
                'e41_emnist_clipinf_0_05_evaluation': 0.05,
                'e41_emnist_clipinf_0_075_evaluation': 0.075,
                # 'e41_clipinf_0.15_evaluation': 0.15
            },
            "PGD_ATTACK": {
            },
            "NO_ATTACK": {
            }
        }
    }
}

DATA_KEYS_CIFAR = {
    "CLIP_DEFENSE": {
        "L2": {
            "BASELINE": 'e41_google_tasks_noconstrain_evaluation',
            "XMAX": 50,
            "XMIN": 0.01,
            "ATTACK": {

            },
            "PGD_ATTACK": {
            },
            "NO_ATTACK": {
            }
        },
        "LINF": {
            "BASELINE": 'e41_google_tasks_noconstrain_evaluation',
            "XMAX": 0.2,
            "XMIN": 0.00004,
            "ATTACK": {

            },
            "PGD_ATTACK": {
            },
            "NO_ATTACK": {
            }
        }
    }
}


# Theming !
def get_grayscale_styles():
    colors = ['0.1', '0.3', '0.6']
    linestyles = ['-', '--', '-']
    return colors, linestyles


COLOR_BENIGN = "#c3ddec"


def get_colorful_styles():
    cmap_1 = matplotlib.cm.get_cmap('Set1')
    cmap_2 = matplotlib.cm.get_cmap('Set2')
    # colors = [cmap_1(i) for i in range(8)]
    colors = []
    colors.extend([cmap_2(i) for i in range(30)])
    # colors = ['#CD4631', '#8B1E3F', '#3C153B', '#89BD9E', '#F9C80E']
    linestyles = ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-',
                  '-', '-', '-', '-', '-', '-', '-', '-', '-', '-']
    return colors, linestyles


def get_large_figsize(fig_width_pt=300.0, golden_mean=None):
    # fig_width_pt = 300.0  # Get this from LaTeX using \showthe
    inches_per_pt = 1.0 / 72.27 * 2  # Convert pt to inches
    if golden_mean is None:
        golden_mean = ((np.math.sqrt(5) - 1.0) / 2.0) * .8  # Aesthetic ratio
    fig_width = fig_width_pt * inches_per_pt  # width in inches
    fig_height = (fig_width * golden_mean)  # height in inches
    fig_size = [fig_width, fig_height / 1.22]
    return fig_height, fig_size, fig_width


def get_progressive_colors(totals=10.0):
    cmap_1 = matplotlib.cm.get_cmap('summer')
    # totals = 10.0
    colors = [cmap_1(i) for i in np.arange(0, 1, 1.0 / totals)]
    # colors = ['#CD4631', '#8B1E3F', '#3C153B', '#89BD9E', '#F9C80E']
    # linestyles = ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-']
    return colors


# colors, linestyles = get_colorful_styles()
colors, linestyles = get_grayscale_styles()

def backup_array(arr, name):
    np.save(os.path.join(plot_data_save_path, name), arr)


def load_backup_array(name):
    return np.load(os.path.join(plot_data_save_path, name + ".npy"))


def cifar_lenet_wr_plot(plotname):
    df = pd.read_csv(os.path.join(plot_data_save_path, 'cifar_lenet_wr_varying.csv'))
    # print(df)
    adv = 'adv_success'
    suc = 'test_accuracy'
    params, fig_size = get_plt_params()

    plt.rcParams.update(params)
    matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    matplotlib.rc('text', usetex=True)

    pdf_pages = PdfPages('./plots/%s' % plotname)

    plt.axes([0.12, 0.32, 0.85, 0.63], frameon=True)
    plt.rc('pdf', fonttype=42)  # IMPORTANT to get rid of Type 3

    f, ax1 = plt.subplots()
    wrs = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    results_adv = [(wrs[i], df[f"run-{i}_evaluation/{adv}"][4]) for i in range(0, 11)]
    results_adv_x, results_adv_y = zip(*results_adv)

    results_ben = [(wrs[i], df[f"run-{i}_evaluation/{suc}"][4]) for i in range(0, 11)]
    results_ben_x, results_ben_y = zip(*results_ben)

    plt.plot(results_adv_x, results_adv_y, '-o', label="Adversarial objective", color=colors[1], linewidth=2)
    plt.plot(results_ben_x, results_ben_y, '-o', label="Benign objective", color=colors[0], linewidth=2)

    # plt.
    # plt.scatter(pgd_compare.values(), [compare_pgd_mean], label="PGD", color=colors[3])

    # print(df[f"e41_clipl2_0_05_noattack_evaluation/{suc}"].last_valid_index())
    # for id, (key, norm) in enumerate(evaluate.items()):
    #     # df.plot(x='Round', y=plot_legend[type], style='o', label=plot_legend[type], color=colors[id], linestyle=linestyles[id], linewidth=2)
    #     plt.plot(norm, df[type], label=key, color=colors[id], linestyle=linestyles[id], linewidth=2)

    plt.xlabel('Weight regularization factor')
    # ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: '%.0fk' % (y * 1e-3)))

    plt.ylabel("Accuracy")
    plt.legend(bbox_to_anchor=(-0.016, 1.00, 1., .102), loc=3, ncol=2, columnspacing=0.75)

    plt.grid(True, linestyle=':', color='0.8', zorder=0)
    F = plt.gcf()
    F.set_size_inches(fig_size)
    pdf_pages.savefig(F, bbox_inches='tight')
    plt.clf()
    pdf_pages.close()

def norm_accuracy_tradeoff_plot(plotname, norm, DATA_KEYS, csv):
    df = pd.read_csv(os.path.join(plot_data_save_path, csv))
    # print(df)
    adv = 'adv_success'
    suc = 'test_accuracy'
    baseline_success = f'{DATA_KEYS["CLIP_DEFENSE"][norm]["BASELINE"]}/{suc}'

    evaluate = DATA_KEYS["CLIP_DEFENSE"][norm]["ATTACK"]
    accuracies = {}
    adversaries = {}

    noattack_compare = DATA_KEYS["CLIP_DEFENSE"][norm]["NO_ATTACK"]
    pgd_compare = DATA_KEYS["CLIP_DEFENSE"][norm]["PGD_ATTACK"]

    last_runs_count = 25

    def calcMean(s, last_valid_index=670):
        avg = s[last_valid_index - last_runs_count:last_valid_index].mean()
        return avg

    for key in evaluate.keys():
        # calculate average accuracy over last 10 runs ?

        avg_eval = calcMean(df[f"{key}/{suc}"])
        avg_adv = calcMean(df[f"{key}/{adv}"])

        accuracies[key] = avg_eval
        adversaries[key] = avg_adv

    # plot_legend = {'e41_clipl2_3_evaluation/adv_success': '3',
    #                'e41_clipl2_5_evaluation/adv_success': '5',
    #                 'femnist_norm_inspect_data_poison_l2_total/mal': 'Mal. (DP)',
    #                'femnist_norm_inspect_scaled_l2_total/mal': 'Mal. (LM, scaled)'}
    #
    params, fig_size = get_plt_params()

    plt.rcParams.update(params)
    matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    matplotlib.rc('text', usetex=True)

    pdf_pages = PdfPages('./plots/%s' % plotname)

    plt.axes([0.12, 0.32, 0.85, 0.63], frameon=True)
    plt.rc('pdf', fonttype=42)  # IMPORTANT to get rid of Type 3

    f, ax1 = plt.subplots()

    keys = evaluate.keys()
    norms = [evaluate[k] for k in keys]
    ben_success = [accuracies[k] for k in keys]
    adv_success = [adversaries[k] for k in keys]

    baseline_mean = calcMean(df[baseline_success])
    noattack_keys = noattack_compare.keys()
    compare_mean = [calcMean(df[f"{a}/{suc}"]) for a in noattack_keys]
    compare_pgd_mean = [calcMean(df[f"{a}/{adv}"], df[f"{a}/{adv}"].last_valid_index()) for a in pgd_compare.keys()]

    plt.plot([0, 1000], [baseline_mean, baseline_mean], label="Baseline (no constraint)", color=colors[0],
             linestyle='--',
             linewidth=2, alpha=0.5)
    # plt.plot([noattack_compare[a] for a in noattack_keys], compare_mean, label="Baseline (no adversary)",
    #          color=colors[2],
    #          linestyle='--', alpha=0.5)

    # print(ben_success)
    # print(norms)

    plt.scatter(norms, ben_success, label="Global objective", color=colors[0], linestyle=linestyles[1], linewidth=2)
    plt.scatter(norms, adv_success, label="Adv. success", color=colors[1], linestyle=linestyles[0], linewidth=2)
    # plt.scatter(pgd_compare.values(), [compare_pgd_mean], label="PGD", color=colors[3])

    # print(df[f"e41_clipl2_0_05_noattack_evaluation/{suc}"].last_valid_index())
    # for id, (key, norm) in enumerate(evaluate.items()):
    #     # df.plot(x='Round', y=plot_legend[type], style='o', label=plot_legend[type], color=colors[id], linestyle=linestyles[id], linewidth=2)
    #     plt.plot(norm, df[type], label=key, color=colors[id], linestyle=linestyles[id], linewidth=2)

    if norm == "L2":
        plt.xlabel('$L_2$-norm')
    else:
        plt.xlabel('$L_{\infty}$-norm')
    plt.xscale('log')
    plt.xlim(DATA_KEYS["CLIP_DEFENSE"][norm]["XMIN"], DATA_KEYS["CLIP_DEFENSE"][norm]["XMAX"])
    # ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: '%.0fk' % (y * 1e-3)))

    plt.ylabel("Accuracy")
    plt.legend(bbox_to_anchor=(-0.016, 1.00, 1., .102), loc=3, ncol=2, columnspacing=0.75)

    plt.grid(True, linestyle=':', color='0.8', zorder=0)
    F = plt.gcf()
    F.set_size_inches(fig_size)
    pdf_pages.savefig(F, bbox_inches='tight')
    plt.clf()
    pdf_pages.close()


def get_plt_params():
    # fig_height, fig_size, fig_width = get_large_figsize()
    #
    # params = {'backend': 'ps',
    #           'axes.labelsize': FONT_SIZE,
    #           'legend.fontsize': FONT_SIZE,
    #           'xtick.labelsize': FONT_SIZE,
    #           'ytick.labelsize': FONT_SIZE,
    #           'font.size': FONT_SIZE,
    #           'figure.figsize': fig_size,
    #           'font.family': 'times new roman'}

    fig_width_pt = 220  # Get this from LaTeX using \showthe
    inches_per_pt = 1.0 / 72.27 * 2  # Convert pt to inches
    golden_mean = ((np.math.sqrt(5) - 1.0) / 2.0) * .8  # Aesthetic ratio
    fig_width = fig_width_pt * inches_per_pt  # width in inches
    fig_height = (fig_width * golden_mean)  # height in inches
    fig_size =  [fig_width, fig_height]

    plt_params = {
        'backend': 'ps',
        'axes.labelsize': 20,
        'legend.fontsize': 16,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'font.size': 18,
        'figure.figsize': fig_size,
        'font.family': 'Times New Roman'
    }

    return plt_params, fig_size


def norm_accuracy_compare_plot(plotname, norm):
    df = pd.read_csv(os.path.join(plot_data_save_path, 'femnist_bounds.csv'))
    LOCAL_DATA_KEYS = {
        "CLIP_DEFENSE": {
            "L2": {
                "BASELINE": 'e41_google_tasks_noconstrain_evaluation',
                "XMAX": 50,
                "XMIN": 0.01,
                "ATTACK": {
                    'e41_clipl2_0_01_evaluation': 0.01,
                    'e41_clipl2_0_1_evaluation': 0.1,
                    'e41_clipl2_0_5_evaluation': 0.5,
                    'e41_clipl2_1_evaluation': 1,
                    'e41_clipl2_3_5_evaluation': 3.5,
                    'e41_clipl2_5_evaluation': 5,
                    'e41_clipl2_10_evaluation': 10
                }
            },
            "LINF": {
                "BASELINE": 'e41_google_tasks_noconstrain_evaluation',
                "XMAX": 0.2,
                "XMIN": 0.001,
                "ATTACK": {
                    'e41_clipinf_0_00100_evaluation': 0.0010,
                    # 'e41_clipinf_0.0015_evaluation': 0.0015,
                    'e41_clipinf_0.005_evaluation': 0.005,
                    'e41_clipinf_0.015_evaluation': 0.015,
                    'e41_clipinf_0.010_evaluation': 0.01,
                    # 'e41_clipinf_0.020_evaluation': 0.02,
                    # 'e41_clipinf_0.025_evaluation': 0.025,
                    'e41_clipinf_0_03_evaluation': 0.03,
                    # 'e41_clipinf_0.15_evaluation': 0.15
                },
            }
        }
    }
    # print(df)
    plot_types_obj = {f"{key}/test_accuracy": val for (key, val) in LOCAL_DATA_KEYS["CLIP_DEFENSE"][norm]["ATTACK"].items()}
    plot_types_mal = {f"{key}/adv_success": val for (key, val) in LOCAL_DATA_KEYS["CLIP_DEFENSE"][norm]["ATTACK"].items()}
    #
    # plot_legend = {'e41_clipl2_3_evaluation/adv_success': '3',
    #                'e41_clipl2_5_evaluation/adv_success': '5',
    #                'femnist_norm_inspect_data_poison_l2_total/mal': 'Mal. (DP)',
    #                'femnist_norm_inspect_scaled_l2_total/mal': 'Mal. (LM, scaled)'}

    pdf_pages = PdfPages('./plots/%s' % plotname)
    params, fig_size = get_plt_params()

    plt.rcParams.update(params)
    matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    matplotlib.rc('text', usetex=True)
    plt.axes([0.12, 0.32, 0.85, 0.63], frameon=True)
    plt.rc('pdf', fonttype=42)  # IMPORTANT to get rid of Type 3

    colors, linestyles = get_colorful_styles()
    colors = get_progressive_colors(len(plot_types_obj.keys()))
    f, ax1 = plt.subplots()

    start = 0
    end = 700

    for id, (type, val) in enumerate(plot_types_obj.items()):
        # df.plot(x='Round', y=plot_legend[type], style='o', label=plot_legend[type], color=colors[id], linestyle=linestyles[id], linewidth=2)
        plt.plot(df.Round[start:end], df[type][start:end], label=val, color=colors[id], linestyle=linestyles[id], linewidth=2)
    for id, (type, val) in enumerate(plot_types_mal.items()):
        # df.plot(x='Round', y=plot_legend[type], style='o', label=plot_legend[type], color=colors[id], linestyle=linestyles[id], linewidth=2)
        plt.plot(df.Round[start:end], df[type][start:end], color=colors[id], linestyle=":", linewidth=2)

    # Additional, custom legend

    plt.xlabel('Round')
    plt.xlim(start, end)
    # ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: '%.0fk' % (y * 1e-3)))

    plt.ylabel("Accuracy")
    # plt.legend(bbox_to_anchor=(1., 0., 1., .102), loc=3, ncol=1, columnspacing=0.75)

    bottom_offset = [0.05, -0.25] if norm is "L2" else [0.20, -0.10]

    # Additional, custom legend
    custom_lines_colors = [Line2D([0], [0], linestyle="-", lw=4, color=colors[id]) for id, _ in enumerate(plot_types_mal.items())]
    custom_lines = [Line2D([0], [0], linestyle="-", lw=4, color=COLOR_GRAY),
                    Line2D([0], [0], linestyle=":", lw=4, color=COLOR_GRAY)]
    leg1 = Legend(ax1, custom_lines_colors, [val for _, val in plot_types_obj.items()],
                 loc='lower left', bbox_to_anchor=(1.02, bottom_offset[0], 1., 1.), title="Bound")
    leg1._legend_box.align = "left"
    leg2 = Legend(ax1, custom_lines, ["Benign", "Malicious"],
                 loc='lower left', bbox_to_anchor=(1.02, bottom_offset[1], 1., 1.))
    ax1.add_artist(leg1)
    ax1.add_artist(leg2)

    plt.grid(True, linestyle=':', color='0.8', zorder=0)
    F = plt.gcf()
    F.set_size_inches(fig_size)
    pdf_pages.savefig(F, bbox_inches='tight')
    plt.clf()
    pdf_pages.close()


def norm_per_round(plotname):
    fig_height, fig_size, fig_width = get_large_figsize()

    pdf_pages = PdfPages('./plots/%s' % plotname)
    params, fig_size = get_plt_params()

    plt.rcParams.update(params)
    matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    matplotlib.rc('text', usetex=True)

    plt.axes([0.12, 0.32, 0.85, 0.63], frameon=True)
    plt.rc('pdf', fonttype=42)  # IMPORTANT to get rid of Type 3

    colors, linestyles = get_colorful_styles()
    f, ax1 = plt.subplots()

    benign = []
    mal = []
    benign_avg = []  # debug

    for i in range(1, 1000, 1):
        # for i in range(1, 4821, 1):
        file = np.load(f'../../experiments_set/norm/normround/round_{i}.npy', allow_pickle=True)
        benign_norms_l2, benign_norms_l1, mal_norms_l2, mal_norms_l1 = file[0], file[1], file[2], file[3]
        benign.append(benign_norms_l2)
        mal.append(mal_norms_l2[0])
        benign_avg.append(np.average(benign_norms_l2))
        # print(f"Reading {i}")

    # plt.boxplot(benign)
    plt.plot(benign_avg, label="Benign (avg)", color=colors[0], linestyle=linestyles[1], linewidth=2)
    plt.plot(mal, label="Malicious", color=colors[1], linestyle=linestyles[1], linewidth=2)

    plt.xlabel('Round')
    # ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: '%.0fk' % (y * 1e-3)))

    plt.ylabel("L2 Norm")
    # plt.yscale("log")
    plt.legend(bbox_to_anchor=(-0.016, 1.00, 1., .102), loc=3, ncol=4, columnspacing=0.75)

    plt.grid(True, linestyle=':', color='0.8', zorder=0)
    F = plt.gcf()
    F.set_size_inches(fig_size)
    pdf_pages.savefig(F, bbox_inches='tight')
    plt.clf()
    pdf_pages.close()


def hypergeometric_distribution(plotname):
    fig_height, fig_size, fig_width = get_large_figsize()

    pdf_pages = PdfPages('./plots/%s' % plotname)
    params, fig_size = get_plt_params()

    plt.rcParams.update(params)
    matplotlib.rc('font', **{'size': 14, 'family': 'serif', 'serif': ['Computer Modern']})
    matplotlib.rc('text', usetex=True)

    plt.axes([0.12, 0.32, 0.85, 0.63], frameon=True)
    plt.rc('pdf', fonttype=42)  # IMPORTANT to get rid of Type 3

    colors, linestyles = get_colorful_styles()
    f, ax1 = plt.subplots()

    def hypergeom_calc(x, frac, total):
        top = scipy.special.comb(total - int(total * frac), x)
        bottom = scipy.special.comb(total, x)
        return 1.0 - (top / bottom)

    total_number_of_weights = 20000
    fractions = [0.1, 0.25, 0.5, 0.75]
    x_values = list(range(1, 101))
    perc = '%'

    for i, f in enumerate(fractions):
        y_values = [hypergeom_calc(x, f, total_number_of_weights) for x in x_values]
        x_values_perc = [float(x) / float(total_number_of_weights) for x in x_values]
        label = f"{(f * 100.0):.0f}\\%"
        plt.plot(x_values_perc, y_values, label=label, color=colors[i], linewidth=2)
    # plt.boxplot(benign)
    # plt.plot(benign_avg, label="Benign (avg)", color=colors[0], linestyle=linestyles[1], linewidth=2)
    # plt.plot(mal, label="Malicious", color=colors[1], linestyle=linestyles[1], linewidth=2)

    plt.xlabel(f'Parameters outside range (total weights = {total_number_of_weights})')
    ax1.xaxis.set_major_formatter(ticker.PercentFormatter())
    # ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: '%.0fk' % (y * 1e-3)))

    plt.ylabel("Detection probability")
    # plt.yscale("log")
    leg = plt.legend(bbox_to_anchor=(-0.016, 1.00, 1., .102), loc=3, ncol=4, columnspacing=0.75,
                     title="Percentage of parameters verified")
    leg._legend_box.align = "left"

    plt.grid(True, linestyle=':', color='0.8', zorder=0)
    F = plt.gcf()
    F.set_size_inches(fig_size)
    pdf_pages.savefig(F, bbox_inches='tight')
    plt.clf()
    pdf_pages.close()


def scaling_factor_adv_success(plotname):
    pdf_pages = PdfPages('./plots/%s' % plotname)
    params, fig_size = get_plt_params()

    plt.rcParams.update(params)
    matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    matplotlib.rc('text', usetex=True)

    plt.axes([0.12, 0.32, 0.85, 0.63], frameon=True)
    plt.rc('pdf', fonttype=42)  # IMPORTANT to get rid of Type 3

    colors, linestyles = get_colorful_styles()
    f, ax1 = plt.subplots()

    NORM_KEY = "l2_total/mal"
    ADV_KEY = "evaluation/adv_success"
    RUNS = {
        # "greencar": {10: 10, 20: 10, 40: 10},
        # "racingstripes": {10: 10, 20: 10, 40: 10},
        # "bgwall": {10: 10, 20: 10, 40: 10}
        "greencar": {40: 10},
        "racingstripes": {40: 10},
        "bgwall": {40: 10}
    }
    SCALING_FACTORS = {
        10: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        20: [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
        40: [1, 5, 9, 13, 17, 23, 27, 31, 35, 40]
    }

    # linestyles = ['-', '--', ':']
    linestyles = ['-', ':']

    ax2 = ax1.twinx()
    for i, (attack, alphas) in enumerate(RUNS.items()):
        for alpha_i, (alpha, cnt) in enumerate(alphas.items()):
            mal_file = pd.read_csv(
                os.path.join(plot_data_save_path, f'l2_comparison_attack/cifar_lenet_minloss_wr_{attack}_{alpha}.csv'))

            mal = []
            for id in range(0, cnt):
                run = f"run-{id}"
                val = mal_file[f"{run}_{ADV_KEY}"][4]
                print(f"{attack} {alpha} {run}: {val}")
                mal.append((mal_file[f"{run}_{NORM_KEY}"][4], mal_file[f"{run}_{ADV_KEY}"][4]))

            mal_norm, malY = zip(*mal)
            ax2.plot(SCALING_FACTORS[alpha], malY, 'o', label=f"{attack},{alpha}", color=colors[i],
                     linestyle=linestyles[alpha_i], linewidth=2)
            ax1.plot(SCALING_FACTORS[alpha], mal_norm, color=colors[i], linestyle=':', linewidth=2)

    ax2.set_ylabel("Adversarial success")
    ax2.set_ylim(0, 1.0)

    ax1.set_xlabel('Scaling factor')
    # ax1.set_xlim(left=1)

    ax1.set_ylabel("$L_2$-norm")
    # plt.legend(bbox_to_anchor=(-0.016, 1.00, 1., .102), loc=3, ncol=4, columnspacing=0.75)

    custom_lines_colors = [Line2D([0], [0], linestyle="-", lw=2, color=colors[0]),
                           Line2D([0], [0], linestyle="-", lw=2, color=colors[1]),
                           Line2D([0], [0], linestyle="-", lw=2, color=colors[2])]
    custom_lines_styles = [Line2D([0], [0], linestyle=ls, lw=2, color=COLOR_GRAY) for ls in linestyles[::-1]]
    custom_benign = [Patch(facecolor=COLOR_BENIGN, label="Benign clients")]
    leg1 = plt.legend(custom_lines_colors, ["Green cars", "Racing stripes", "Background wall"],
                      bbox_to_anchor=(1.12, 0.46, 1., .102), loc=3, ncol=1, columnspacing=0.75,
                      title="Attack")
    leg2 = plt.legend(custom_lines_styles, ["Distance", "Accuracy"],
                      bbox_to_anchor=(1.12, 0.16, 1., .102), loc=3, ncol=1, columnspacing=0.75,
                      )
    # leg3 = plt.legend(handles=custom_benign, bbox_to_anchor=(1.12, -0.26, 1., .102), loc=3, ncol=1, columnspacing=0.75,
    #                   )
    leg1._legend_box.align = "left"
    leg2._legend_box.align = "left"
    # leg3._legend_box.align = "left"
    ax2.add_artist(leg1)
    ax2.add_artist(leg2)
    # ax2.add_artist(leg3)

    # plt.title("Comparison of $L_2$-norm of attacks under different participation rates", y=1.04, fontsize=FONT_SIZE)
    plt.grid(True, linestyle=':', color='0.8', zorder=0)
    F = plt.gcf()
    F.set_size_inches(fig_size)
    pdf_pages.savefig(F, bbox_inches='tight')
    plt.clf()
    pdf_pages.close()

def accuracy_pgd(plotname):
    pdf_pages = PdfPages('./plots/%s' % plotname)
    params, fig_size = get_plt_params()

    plt.rcParams.update(params)
    matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    matplotlib.rc('text', usetex=True)

    plt.axes([0.12, 0.32, 0.85, 0.63], frameon=True)
    plt.rc('pdf', fonttype=42)  # IMPORTANT to get rid of Type 3

    colors, linestyles = get_colorful_styles()
    f, ax1 = plt.subplots()

    df_pgd = pd.read_csv(os.path.join(plot_data_save_path, f'cifar_lenet_pgd.csv'))

    df_baseline = pd.read_csv(os.path.join(plot_data_save_path, 'constant_attack_lenet_bound_plot.csv'))
    baseline_noclip = "cifar_lenet_train_noattack_clip_100_evaluation/test_accuracy"
    baseline_clip = "cifar_lenet_train_noattack_clip_100_evaluation/test_accuracy"

    runs = {
        "run-0": "PGD Attack ($\gamma =5$)",
        "run-1": "PGD Attack ($\gamma =25$)",
        "run-2": "PGD Attack ($\gamma =40$)"
    }

    linestyles = ["-", ":"]

    plt.plot(df_baseline["Round"][:500], df_baseline[baseline_noclip][:500], color=colors[0], linestyle=linestyles[0],
             linewidth=2)
    # plt.plot(df_baseline["Round"][:500], df_baseline[baseline_clip][:500], color=colors[1], linestyle=linestyles[0],
    #          linewidth=2)

    for i, (run, scale) in enumerate(runs.items()):
        plt.plot(df_pgd["Round"], df_pgd[f"{run}_evaluation/test_accuracy"], color=colors[i+1], linestyle=linestyles[0], linewidth=2)
        plt.plot(df_pgd["Round"], df_pgd[f"{run}_evaluation/adv_success"], color=colors[i+1], linestyle=linestyles[1], linewidth=2)

    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0, 0.6)

    ax1.set_xlabel("Round")
    # ax1.set_xlim(left=1)

    # plt.legend(bbox_to_anchor=(-0.016, 1.00, 1., .102), loc=3, ncol=4, columnspacing=0.75)

    run_type_labels = ["Baseline"]
    run_type_labels.extend(list(runs.values()))

    custom_lines_colors = [Line2D([0], [0], linestyle="-", lw=2, color=colors[0]),
                           Line2D([0], [0], linestyle="-", lw=2, color=colors[1]),
                           Line2D([0], [0], linestyle="-", lw=2, color=colors[2]),
                           Line2D([0], [0], linestyle="-", lw=2, color=colors[3]),
                           Line2D([0], [0], linestyle="-", lw=2, color=colors[4])]
    custom_lines_styles = [Line2D([0], [0], linestyle=ls, lw=2, color=COLOR_GRAY) for ls in linestyles]
    leg1 = plt.legend(custom_lines_colors, run_type_labels,
                      bbox_to_anchor=(1., 0.43, 1., .102), loc=3, ncol=1, columnspacing=0.75)
    leg2 = plt.legend(custom_lines_styles, ["Benign objective", "Malicious objective"],
                      bbox_to_anchor=(1., 0.13, 1., .102), loc=3, ncol=1, columnspacing=0.75,
                      )
    # leg3 = plt.legend(handles=custom_benign, bbox_to_anchor=(1.12, -0.26, 1., .102), loc=3, ncol=1, columnspacing=0.75,
    #                   )
    leg1._legend_box.align = "left"
    leg2._legend_box.align = "left"
    # leg3._legend_box.align = "left"
    ax1.add_artist(leg1)
    ax1.add_artist(leg2)
    # ax2.add_artist(leg3)

    # plt.title("Comparison of $L_2$-norm of attacks under different participation rates", y=1.04, fontsize=FONT_SIZE)
    plt.grid(True, linestyle=':', color='0.8', zorder=0)
    F = plt.gcf()
    F.set_size_inches(fig_size)
    pdf_pages.savefig(F, bbox_inches='tight')
    plt.clf()
    pdf_pages.close()

def endtoend_timing_bar(plotname, bound):
    timings = {
        "MNIST_CONV": {
            "label": "MNIST ConvNN \n (19166 param.)",
            "plain": 5.604241216,
            "range": {
                "naive": 278.45,
                "optim": 86.17
            },
            "l2": {
                "naive": 335.77,  # 339.669 seconds, 336
                "optim": 38.50
            }
        },
        "CIFAR_LENET": {
            "label": "CIFAR10 LeNet \n (62006 param.)",
            "plain": 7.31,
            "range": {
                "naive": 660.3487,  # 660.3487005233765 per round
                "optim": 293.35  # 323.424 per round
            },
            "l2": {
                "naive": 801.80,  # TIMING: running now ?? too big to transfer
                "optim": 120.4801153  # TIMING: todo... subspace
            }
        }
    }

    pdf_pages = PdfPages('./plots/%s' % plotname)
    params, fig_size = get_plt_params()

    plt.rcParams.update(params)
    matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    matplotlib.rc('text', usetex=True)

    plt.axes([0.12, 0.32, 0.85, 0.63], frameon=True)
    plt.rc('pdf', fonttype=42)  # IMPORTANT to get rid of Type 3

    colors, linestyles = get_colorful_styles()
    f, ax1 = plt.subplots()

    labels = [x["label"] for (_, x) in timings.items()]

    x = np.arange(len(labels))  # the label locations
    width = 0.15  # the width of the bars

    plt.bar(x - width, [x["plain"] for (_, x) in timings.items()], width, color=colors[0], label="Plain")
    plt.bar(x, [x[bound]["optim"] for (_, x) in timings.items()], width, color=colors[2], label="Optimized")
    plt.bar(x + width, [x[bound]["naive"] for (_, x) in timings.items()], width, color=colors[1], label="Na\\\"{i}ve")

    plt.title("Time per round")
    plt.ylabel("Time (seconds)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)

    plt.legend()

    # plt.title("Comparison of $L_2$-norm of attacks under different participation rates", y=1.04, fontsize=FONT_SIZE)
    plt.grid(True, linestyle=':', color='0.8', zorder=0)
    F = plt.gcf()
    F.set_size_inches(fig_size)
    pdf_pages.savefig(F, bbox_inches='tight')
    plt.clf()
    pdf_pages.close()


def norm_distribution_benign(plotname):
    pdf_pages = PdfPages('./plots/%s' % plotname)
    params, fig_size = get_plt_params()

    plt.rcParams.update(params)
    # matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    # matplotlib.rc('text', usetex=True)

    plt.axes([0.12, 0.32, 0.85, 0.63], frameon=True)
    plt.rc('pdf', fonttype=42)  # IMPORTANT to get rid of Type 3

    colors, linestyles = get_colorful_styles()
    f, ax1 = plt.subplots()

    NORM_KEY = "l2_total/mal"
    ADV_KEY = "evaluation/adv_success"
    RUNS = {
        "greencar": {10: 10, 40: 9},
        "racingstripes": {10: 10, 40: 7},
        "bgwall": {10: 10, 40: 7}
    }

    linestyles = ['-', '--']
    benign_avg = []  # debug
    for i in [6]:
        file = np.load(f'../../experiments_set/cifar_lenet/cifar_lenet_noniid_norms_oldlr/norms/round_{i}.npy',
                       allow_pickle=True)
        # file = np.load(f'../../experiments_set/norm/normround/round_{i}.npy', allow_pickle=True)
        benign_norms_l2, benign_norms_l1, mal_norms_l2, mal_norms_l1 = file[0], file[1], file[2], file[3]
        sns.distplot(benign_norms_l2, hist=False, kde=True,
                     kde_kws={'shade': True, 'linewidth': 0}, ax=ax1)

    ax2 = ax1.twinx()
    for i, (attack, alphas) in enumerate(RUNS.items()):
        for alpha_i, (alpha, cnt) in enumerate(alphas.items()):
            mal_file = pd.read_csv(
                os.path.join(plot_data_save_path, f'l2_comparison_attack/cifar_lenet_minloss_wr_{attack}_{alpha}.csv'))

            mal = []
            for id in range(0, cnt):
                run = f"run-{id}"
                val = mal_file[f"{run}_{ADV_KEY}"][4]
                print(f"{attack} {alpha} {run}: {val}")
                mal.append((mal_file[f"{run}_{NORM_KEY}"][4], mal_file[f"{run}_{ADV_KEY}"][4]))

            malX, malY = zip(*mal)
            ax2.plot(malX, malY, 'o', label=f"{attack},{alpha}", color=colors[i], linestyle=linestyles[alpha_i],
                     linewidth=2)

    ax2.set_ylabel("Adversarial success")
    ax2.set_ylim(0, 1.0)

    ax1.set_xlabel('$L_2$-norm')

    # ax1.set_ylabel("Percentage of benign users")
    ax1.set_ylabel("Density (KDE)")

    # plt.legend(bbox_to_anchor=(-0.016, 1.00, 1., .102), loc=3, ncol=4, columnspacing=0.75)

    custom_lines_colors = [Line2D([0], [0], linestyle="-", lw=2, color=colors[0]),
                           Line2D([0], [0], linestyle="-", lw=2, color=colors[1]),
                           Line2D([0], [0], linestyle="-", lw=2, color=colors[2])]
    custom_lines_styles = [Line2D([0], [0], linestyle=ls, lw=2, color=COLOR_GRAY) for ls in linestyles[::-1]]
    custom_benign = [Patch(facecolor=COLOR_BENIGN, label="Benign clients")]
    leg1 = plt.legend(custom_lines_colors, ["Green cars", "Racing stripes", "Background wall"],
                      bbox_to_anchor=(1.12, 0.46, 1., .102), loc=3, ncol=1, columnspacing=0.75,
                      title="Attack")
    leg2 = plt.legend(custom_lines_styles, ["$2.5\\%$", "$10\\%$"],
                      bbox_to_anchor=(1.12, -0.08, 1., .102), loc=3, ncol=1, columnspacing=0.75,
                      title="$\\alpha$")
    leg3 = plt.legend(handles=custom_benign, bbox_to_anchor=(1.12, -0.26, 1., .102), loc=3, ncol=1, columnspacing=0.75,
                      )
    leg1._legend_box.align = "left"
    leg2._legend_box.align = "left"
    leg3._legend_box.align = "left"
    ax2.add_artist(leg1)
    ax2.add_artist(leg2)
    ax2.add_artist(leg3)

    # plt.title("Comparison of $L_2$-norm of attacks under different participation rates", y=1.04, fontsize=FONT_SIZE)
    plt.grid(True, linestyle=':', color='0.8', zorder=0)
    F = plt.gcf()
    F.set_size_inches(fig_size)
    pdf_pages.savefig(F, bbox_inches='tight')
    plt.clf()
    pdf_pages.close()


def norm_distribution_benign_clients_multiround(plotname):
    pdf_pages = PdfPages('./plots/%s' % plotname)
    params, fig_size = get_plt_params()


    del matplotlib.font_manager.weight_dict['roman']
    matplotlib.font_manager._rebuild()

    plt.rcParams.update(params)
    # matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    # matplotlib.rc('text', usetex=True)
    # matplotlib.rc('axes', axisbelow=True)

    plt.axes([0.12, 0.32, 0.85, 0.63], frameon=True)
    plt.rc('pdf', fonttype=42)  # IMPORTANT to get rid of Type 3

    f, ax1 = plt.subplots()

    # round_num = 1
    # benign_norms_l2, benign_norms_l1 = [], []
    #
    # for round_num in range(1, 11):
    #     file = np.load(f'../../experiments_set/cifar_lenet/dist_iid/norms/round_{round_num}.npy',
    #                allow_pickle=True)
    #     benign_norms_l2.extend(file[0])
    #     benign_norms_l1.extend(file[1])
    # # file = np.load(f'../../experiments_set/norm/normround/round_{i}.npy', allow_pickle=True)
    # #     benign_norms_l2, benign_norms_l1 = file[0], file[1]
    #
    # print("IID", benign_norms_l2)

    # sns.distplot(benign_norms_l2, hist=True, kde=True,
    #              kde_kws={'shade': True, 'linewidth': 0}, ax=ax1, label=f"IID")

    benign_norms_l2, benign_norms_l1 = [], []

    for round_num in range(1, 11):
        file = np.load(f'./data/cifar_lenet/dist_noniid/round_{round_num}.npy',
                   allow_pickle=True)
        benign_norms_l2.extend(file[0])
        benign_norms_l1.extend(file[1])
    # file = np.load(f'../../experiments_set/norm/normround/round_{i}.npy', allow_pickle=True)
    # benign_norms_l2, benign_norms_l1 = file[0], file[1]
    print("NonIID", benign_norms_l2)
    sns.distplot(benign_norms_l2, hist=False, kde=True,
                 kde_kws={'shade': True, 'linewidth': 0, 'alpha': 1}, ax=ax1, color=colors[2])
    # plt.axvline(x=1.9, ymin=0, ymax=1, label="Norm bound (1.9)", linestyle="--", color=colors[1])

    ax1.set_xlabel('$L_2$-norm')
    ax1.set_axisbelow(True)
    # ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: '%.0fk' % (y * 1e-3)))

    ax1.set_ylabel("Density (KDE)")
    # plt.xscale("log")
    # plt.yscale("log")
    # plt.legend(bbox_to_anchor=(-0.016, 1.00, 1., .102), loc=3, ncol=4, columnspacing=0.75)

    ax1.grid(True, axis="y", linestyle=':', color='0.6', zorder=0, linewidth=1.2)
    F = plt.gcf()
    F.set_size_inches(fig_size)
    pdf_pages.savefig(F, bbox_inches='tight')
    plt.clf()
    pdf_pages.close()


def norm_distribution_benign_overtime(plotname):
    pdf_pages = PdfPages('./plots/%s' % plotname)
    params, fig_size = get_plt_params()

    plt.rcParams.update(params)
    matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    matplotlib.rc('text', usetex=True)

    plt.axes([0.12, 0.32, 0.85, 0.63], frameon=True)
    plt.rc('pdf', fonttype=42)  # IMPORTANT to get rid of Type 3

    f, ax1 = plt.subplots()

    benign = []
    mal = []
    benign_avg = []  # debug

    round_step = 1000

    p_colors = get_progressive_colors()

    for i in range(1, 100, 10):
        # for i in range(1, 4821, round_step):
        # for i in range(1, 4821, 1):
        file = np.load(f'../../experiments_set/norm/normround/round_{i}.npy', allow_pickle=True)
        benign_norms_l2, benign_norms_l1, mal_norms_l2, mal_norms_l1 = file[0], file[1], file[2], file[3]
        benign.append(benign_norms_l2)
        mal.append(mal_norms_l2[0])
        benign_avg.append(np.average(benign_norms_l2))
        # print(f"Reading {i}")

    # plt.boxplot(benign)
    # plt.plot(benign_avg, label="Benign (avg)", color=colors[0], linestyle=linestyles[1], linewidth=2)
    # plt.plot(mal, label="Malicious", color=colors[1], linestyle=linestyles[1], linewidth=2)
    # print(benign)
    for i, b in enumerate(benign):
        sns.distplot(b, hist=False,  # For histogram
                     kde=True,
                     kde_kws={'shade': True, 'linewidth': 0, 'clip': (0.0, 7.0)},
                     label=f"Round {(i) * round_step + 1}", color=p_colors[i])

    plt.xlabel('$L_2$-norm')
    # ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: '%.0fk' % (y * 1e-3)))

    plt.ylabel("Percentage of benign users")
    # plt.yscale("log")
    # plt.ylim(0, 7)
    plt.legend(bbox_to_anchor=(-0.016, 1.00, 1., .102), loc=3, ncol=4, columnspacing=0.75)

    plt.grid(True, linestyle=':', color='0.8', zorder=0)
    F = plt.gcf()
    F.set_size_inches(fig_size)
    pdf_pages.savefig(F, bbox_inches='tight')
    plt.clf()
    pdf_pages.close()


def squarerandproof_log_plot(plotname):
    df = pd.read_csv(os.path.join(plot_data_save_path, 'microbench_squarerandproof32bit.csv'))
    # print(df)
    plot_types = ['baseline_create',
                  'square_create']
    plot_legend = {'baseline_create': 'Randomness Proof',
                   'square_create': 'Squared Randomness Proof'}

    pdf_pages = PdfPages('./plots/%s' % plotname)
    params, fig_size = get_plt_params()

    plt.rcParams.update(params)
    matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    matplotlib.rc('text', usetex=True)
    plt.rcParams['axes.titlepad'] = 50

    plt.axes([0.12, 0.32, 0.85, 0.63], frameon=True)
    plt.rc('pdf', fonttype=42)  # IMPORTANT to get rid of Type 3

    colors, linestyles = get_colorful_styles()
    plt.subplots()

    for id, type in enumerate(plot_types):
        # df.plot(x='Round', y=plot_legend[type], style='o', label=plot_legend[type], color=colors[id], linestyle=linestyles[id], linewidth=2)
        plt.semilogx(df.parameters, df[type] / 1000.0, '-o', basex=2, label=plot_legend[type], color=colors[id],
                     linestyle=linestyles[id], linewidth=2)

    plt.xlabel('Parameters')
    plt.title("Create Randomness Proof (32-bit precision)")
    # ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: '%.0fk' % (y * 1e-3)))

    plt.ylabel("Time (seconds)")
    # plt.yscale("log")
    plt.legend(bbox_to_anchor=(-0.016, .98, 1., .102), loc=3, ncol=4, columnspacing=0.75)

    plt.grid(True, linestyle=':', color='0.8', zorder=0)
    F = plt.gcf()
    F.set_size_inches(fig_size)
    pdf_pages.savefig(F, bbox_inches='tight')
    plt.clf()
    pdf_pages.close()


def squarerandproof_verify_log_plot(plotname):
    df = pd.read_csv(os.path.join(plot_data_save_path, 'microbench_squarerandproof32bit.csv'))
    # print(df)
    plot_types = ['baseline_verify',
                  'square_verify']
    plot_legend = {'baseline_verify': 'Randomness Proof',
                   'square_verify': 'Squared Randomness Proof'}

    pdf_pages = PdfPages('./plots/%s' % plotname)
    params, fig_size = get_plt_params()

    plt.rcParams.update(params)
    matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    matplotlib.rc('text', usetex=True)

    plt.axes([0.12, 0.32, 0.85, 0.63], frameon=True)
    plt.rc('pdf', fonttype=42)  # IMPORTANT to get rid of Type 3

    plt.rcParams['axes.titlepad'] = 50

    colors, linestyles = get_colorful_styles()
    plt.subplots()

    for id, type in enumerate(plot_types):
        # df.plot(x='Round', y=plot_legend[type], style='o', label=plot_legend[type], color=colors[id], linestyle=linestyles[id], linewidth=2)
        plt.semilogx(df.parameters, df[type] / 1000.0, '-o', basex=2, label=plot_legend[type], color=colors[id],
                     linestyle=linestyles[id], linewidth=2)

    plt.xlabel('Parameters')
    plt.title("Verify Randomness Proof (32-bit Precision)")
    # ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: '%.0fk' % (y * 1e-3)))

    plt.ylabel("Time (seconds)")
    # plt.yscale("log")
    plt.legend(bbox_to_anchor=(-0.016, 0.98, 1., .102), loc=3, ncol=4, columnspacing=0.75)

    plt.grid(True, linestyle=':', color='0.8', zorder=0)
    F = plt.gcf()
    F.set_size_inches(fig_size)
    pdf_pages.savefig(F, bbox_inches='tight')
    plt.clf()
    pdf_pages.close()


def l2proof_plots():
    # print(df)
    lengths = [32]
    ranges = [32, 16, 8]
    actions = ['create', 'verify']
    for l in lengths:
        df = pd.read_csv(os.path.join(plot_data_save_path, f'microbench_l2proof{l}bit.csv'))
        for action in actions:
            plt.figure()
            plotname = f"microbenchmark_l2_{action}_{l}bit.pdf"
            pdf_pages = PdfPages('./plots/%s' % plotname)
            params, fig_size = get_plt_params()

            plt.rcParams.update(params)
            matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
            matplotlib.rc('text', usetex=True)

            plt.axes([0.12, 0.32, 0.85, 0.63], frameon=True)
            plt.rc('pdf', fonttype=42)  # IMPORTANT to get rid of Type 3

            f, ax1 = plt.subplots()

            for id, range in enumerate(ranges):
                type_baseline = f"baseline_r{range}_{action}"
                type_l2 = f"l2_r{range}_{action}"
                # print(df[type_baseline])
                plt.semilogx(df.parameters, df[type_baseline] / 1000.0, '-o', basex=2, color=colors[id],
                             linestyle="--", linewidth=2)
                plt.semilogx(df.parameters, df[type_l2] / 1000.0, '-o', basex=2,
                             color=colors[id],
                             linestyle="-", linewidth=2)

            plt.xlabel('Parameters')
            plt.title(f"{action.capitalize()} Range Proof ({l}-bit precision)")
            # ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: '%.0fk' % (y * 1e-3)))

            plt.ylabel("Time (seconds)")
            # plt.yscale("log")
            # plt.legend(bbox_to_anchor=(-0.016, 1.00, 1., .102), loc=3, ncol=4, columnspacing=0.75)
            # Additional, custom legend
            custom_lines_colors = [Line2D([0], [0], linestyle="-", lw=2, color=colors[0]),
                                   Line2D([0], [0], linestyle="-", lw=2, color=colors[1]),
                                   Line2D([0], [0], linestyle="-", lw=2, color=colors[2])]
            custom_lines_styles = [Line2D([0], [0], linestyle="-", lw=2, color=COLOR_GRAY),
                                   Line2D([0], [0], linestyle="--", lw=2, color=COLOR_GRAY)]
            leg1 = plt.legend(custom_lines_colors, ["32-bit", "16-bit", "8-bit"],
                              bbox_to_anchor=(1., 0.50, 1., .102), loc=3, ncol=1, columnspacing=0.75, title="Range")
            leg1._legend_box.align = "left"
            leg2 = plt.legend(custom_lines_styles, ["$L_2$", "$L_\\infty$"], bbox_to_anchor=(1., 0.11, 1., .102), loc=3,
                              title="Norm",
                              ncol=1, columnspacing=0.75)
            leg2._legend_box.align = "left"
            ax1.add_artist(leg1)
            ax1.add_artist(leg2)

            plt.grid(True, linestyle=':', color='0.8', zorder=0)
            F = plt.gcf()
            F.set_size_inches(fig_size)
            pdf_pages.savefig(F, bbox_inches='tight')
            plt.clf()
            pdf_pages.close()


def l2proof_flexible_case():
    df = pd.read_csv(os.path.join(plot_data_save_path, 'microbench_l2proof32bit.csv'))
    # print(df)

    actions = ['create', 'verify']
    for action in actions:
        plotname = f"microbenchmark_l2_{action}_flexible.pdf"
        pdf_pages = PdfPages('./plots/%s' % plotname)
        params, fig_size = get_plt_params()

        plt.rcParams.update(params)
        matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
        matplotlib.rc('text', usetex=True)

        plt.axes([0.12, 0.32, 0.85, 0.63], frameon=True)
        plt.rc('pdf', fonttype=42)  # IMPORTANT to get rid of Type 3

        plt.subplots()

        plt.semilogx(df.parameters, df[f"baseline_r32_{action}"] / 1000.0, '-o', basex=2, label="$L_\\infty$",
                     color=colors[1],
                     linestyle=linestyles[0], linewidth=2)
        plt.semilogx(df.parameters, df[f"l2_r32_p8_{action}"] / 1000.0, '-o', basex=2, label="$L_2$", color=colors[0],
                     linestyle=linestyles[0], linewidth=2)

        plt.xlabel('Parameters')
        plt.title(f"{action.capitalize()} Norm Bound Proof (32-bit range, 8-bit parameters)")

        # ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: '%.0fk' % (y * 1e-3)))

        plt.ylabel("Time (seconds)")
        # plt.yscale("log")
        leg = plt.legend(bbox_to_anchor=(1., 0.61, 1., .102), loc=3, ncol=1, columnspacing=0.75, title="Norm")
        leg._legend_box.align = "left"

        plt.grid(True, linestyle=':', color='0.8', zorder=0)
        F = plt.gcf()
        F.set_size_inches(fig_size)
        pdf_pages.savefig(F, bbox_inches='tight')
        plt.clf()
        pdf_pages.close()


def microbench_proof_arbitrary_ranges():
    df = pd.read_csv(os.path.join(plot_data_save_path, 'microbenchmark_arbitraryrange.csv'))
    # print(df)

    actions = ['create', 'verify']
    for action in actions:
        plotname = f"microbenchmark_arbitraryrange_{action}.pdf"
        pdf_pages = PdfPages('./plots/%s' % plotname)
        params, fig_size = get_plt_params()

        plt.rcParams.update(params)
        matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
        matplotlib.rc('text', usetex=True)

        plt.axes([0.12, 0.32, 0.85, 0.63], frameon=True)
        plt.rc('pdf', fonttype=42)  # IMPORTANT to get rid of Type 3

        plt.subplots()

        plt.semilogx(df.parameters, df[f"linf_{action}"] / 1000.0, '-o', basex=2, label="$L_\\infty$", color=colors[1],
                     linestyle=linestyles[0], linewidth=2)
        plt.semilogx(df.parameters, df[f"l2_{action}"] / 1000.0, '-o', basex=2, label="$L_2$", color=colors[0],
                     linestyle=linestyles[0], linewidth=2)

        plt.xlabel('Parameters')
        plt.title(f"{action.capitalize()} Arbitrary Range (32-bit range, 32-bit parameters)")

        # ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: '%.0fk' % (y * 1e-3)))

        plt.ylabel("Time (seconds)")
        # plt.yscale("log")
        leg = plt.legend(loc=2, ncol=1, columnspacing=0.75, title="Norm")
        leg._legend_box.align = "left"

        plt.grid(True, linestyle=':', color='0.8', zorder=0)
        F = plt.gcf()
        F.set_size_inches(fig_size)
        pdf_pages.savefig(F, bbox_inches='tight')
        plt.clf()
        pdf_pages.close()


def inspect_norm_plot_lm_scale(plotname):
    df = pd.read_csv(os.path.join(plot_data_save_path, 'femnist_norm_inspect-output.csv'))
    # print(df)
    # print("HEY")
    plot_types = ['femnist_norm_inspect_l2_total/benign',
                  # 'femnist_norm_inspect_l2_total/mal',
                  # 'femnist_norm_inspect_data_poison_l2_total/mal',
                  'femnist_norm_inspect_scaled_l2_total/mal']
    plot_legend = {'femnist_norm_inspect_l2_total/benign': 'Benign',
                   'femnist_norm_inspect_l2_total/mal': 'Mal. (LM)',
                   'femnist_norm_inspect_data_poison_l2_total/mal': 'Mal. (DP)',
                   'femnist_norm_inspect_scaled_l2_total/mal': 'Mal. (SP, scaled by $\gamma=30$)'}

    pdf_pages = PdfPages('./plots/%s' % plotname)
    params, fig_size = get_plt_params()

    plt.rcParams.update(params)
    matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    matplotlib.rc('text', usetex=True)

    plt.axes([0.12, 0.32, 0.85, 0.63], frameon=True)
    plt.rc('pdf', fonttype=42)  # IMPORTANT to get rid of Type 3

    colors, linestyles = get_colorful_styles()
    f, ax1 = plt.subplots()

    for id, type in enumerate(plot_types):
        # df.plot(x='Round', y=plot_legend[type], style='o', label=plot_legend[type], color=colors[id], linestyle=linestyles[id], linewidth=2)
        plt.plot(df.Round, df[type], label=plot_legend[type], color=colors[id], linestyle=linestyles[id], linewidth=2)

    plt.xlabel('Round')
    # ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: '%.0fk' % (y * 1e-3)))

    plt.ylabel("Update L2-norm")
    plt.yscale("log")
    plt.legend(bbox_to_anchor=(-0.016, 1.00, 1., .102), loc=3, ncol=4, columnspacing=0.75)

    plt.grid(True, linestyle=':', color='0.8', zorder=0)
    F = plt.gcf()
    F.set_size_inches(fig_size)
    pdf_pages.savefig(F, bbox_inches='tight')
    plt.clf()
    pdf_pages.close()


def inspect_norm_plot(plotname):
    df = pd.read_csv(os.path.join(plot_data_save_path, 'femnist_norm_inspect-output.csv'))
    # print(df)
    plot_types = ['femnist_norm_inspect_l2_total/benign',
                  'femnist_norm_inspect_l2_total/mal',
                  'femnist_norm_inspect_data_poison_l2_total/mal',
                  # 'femnist_norm_inspect_scaled_l2_total/mal'
                  ]
    plot_legend = {'femnist_norm_inspect_l2_total/benign': 'Benign',
                   'femnist_norm_inspect_l2_total/mal': 'Mal. (LM)',
                   'femnist_norm_inspect_data_poison_l2_total/mal': 'Mal. (DP)',
                   'femnist_norm_inspect_scaled_l2_total/mal': 'Mal. (Segment poisoning, scaled by $\gamma=30$)'}

    fig_height, fig_size, fig_width = get_large_figsize()

    params, fig_size = get_plt_params()

    pdf_pages = PdfPages('./plots/%s' % plotname)

    plt.rcParams.update(params)
    plt.axes([0.12, 0.32, 0.85, 0.63], frameon=True)
    plt.rc('pdf', fonttype=42)  # IMPORTANT to get rid of Type 3

    colors, linestyles = get_colorful_styles()
    f, ax1 = plt.subplots()

    for id, type in enumerate(plot_types):
        # df.plot(x='Round', y=plot_legend[type], style='o', label=plot_legend[type], color=colors[id], linestyle=linestyles[id], linewidth=2)
        plt.plot(df.Round, df[type], label=plot_legend[type], color=colors[id], linestyle=linestyles[id], linewidth=2)

    plt.xlabel('Round')
    # ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: '%.0fk' % (y * 1e-3)))

    plt.ylabel("Update L2-norm")
    plt.legend(bbox_to_anchor=(-0.016, 1.00, 1., .102), loc=3, ncol=4, columnspacing=0.75)

    plt.grid(True, linestyle=':', color='0.8', zorder=0)
    F = plt.gcf()
    F.set_size_inches(fig_size)
    pdf_pages.savefig(F, bbox_inches='tight')
    plt.clf()
    pdf_pages.close()


def modelreplacement_cifar_plot(plotname):
    df = pd.read_csv(os.path.join(plot_data_save_path, 'e44_cifar_resnet.csv'))
    df["Round"] = df["Round"].apply(lambda x: x - 5)
    # print(df)
    plot_types = [
        'e44_cifar_attack_400_0.0001_full_evaluation',
                  'e44_cifar_attack_400_0.0001_full_greencars_evaluation',
                  'e44_cifar_resnet_racing_stripes_evaluation'
                  ]

    params, fig_size = get_plt_params()

    pdf_pages = PdfPages('./plots/%s' % plotname)

    plt.rcParams.update(params)
    plt.axes([0.12, 0.32, 0.85, 0.63], frameon=True)
    plt.rc('pdf', fonttype=42)  # IMPORTANT to get rid of Type 3

    plt.rcParams.update(params)
    matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    matplotlib.rc('text', usetex=True)

    colors, linestyles = get_colorful_styles()
    f, ax1 = plt.subplots()

    legend_custom = {
        'adv_success': 'Malicious objective',
        'test_accuracy': 'Benign objective'
    }
    linestyles_custom = {
        'adv_success': '-.',
        'test_accuracy': '-'
    }
    colors_custom = {
        'e44_cifar_attack_400_0.0001_full_greencars_evaluation': colors[0],
        'e44_cifar_resnet_racing_stripes_evaluation': colors[1],
        'e44_cifar_attack_400_0.0001_full_evaluation': colors[2]  # add colors 1
    }

    for id, type in enumerate(plot_types):
        for suffix in ['adv_success', 'test_accuracy']:
            # print(f"{type}/{suffix}")
            # print(df[f"{type}/{suffix}"])
            # df.plot(x='Round', y=plot_legend[type], style='o', label=plot_legend[type], color=colors[id], linestyle=linestyles[id], linewidth=2)
            plt.plot(df.Round, df[f"{type}/{suffix}"], label=legend_custom[suffix], color=colors_custom[type],
                     linestyle=linestyles_custom[suffix], linewidth=2)

    plt.xlabel('Round')
    # ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: '%.0fk' % (y * 1e-3)))

    plt.ylabel("Accuracy")
    plt.ylim(ymin=0, ymax=1.0)
    plt.xlim(xmin=-5, xmax=300)
    start, end = ax1.get_xlim()
    xticks = np.arange(0, end + 1, 100)
    # np.insert(xticks, 0, -5, axis=0)
    ax1.xaxis.set_ticks(xticks)

    # Additional, custom legend
    custom_lines_colors = [Line2D([0], [0], linestyle="-", lw=2, color=colors[2]),
                           Line2D([0], [0], linestyle="-", lw=2, color=colors[1]),
                           Line2D([0], [0], linestyle="-", lw=2, color=colors[0])]
    custom_lines_styles = [Line2D([0], [0], linestyle="-", lw=2, color=COLOR_GRAY),
                           Line2D([0], [0], linestyle="-.", lw=2, color=COLOR_GRAY)]
    leg1 = plt.legend(custom_lines_colors, ["Background wall", "Racing stripes", "Green cars"],
                      bbox_to_anchor=(1., 0.57, 1., .102), loc=3, ncol=1, columnspacing=0.75)
    leg2 = plt.legend(custom_lines_styles, ["Benign objective", "Malicious objective"],
                      bbox_to_anchor=(1., 0.27, 1., .102), loc=3, ncol=1, columnspacing=0.75)
    ax1.add_artist(leg1)
    ax1.add_artist(leg2)

    plt.grid(True, linestyle=':', color='0.8', zorder=0)
    F = plt.gcf()
    F.set_size_inches(fig_size)
    pdf_pages.savefig(F, bbox_inches='tight')
    plt.clf()
    pdf_pages.close()


def modelreplacement_subspacepoisoning_attack_compare(plotname):
    df = pd.read_csv(os.path.join(plot_data_save_path, 'e44_cifar_resnet.csv'))
    df["Round"] = df["Round"].apply(lambda x: x - 5)

    # print(df)
    plot_types = [
        'e44_cifar_resnet_racing_stripes_evaluation',  #
        'resnet_cifar_greencars_lm_cmp_evaluation'  # It says green cars but it is actually racing stripes !!
    ]

    params, fig_size = get_plt_params()

    pdf_pages = PdfPages('./plots/%s' % plotname)

    plt.rcParams.update(params)
    plt.axes([0.12, 0.32, 0.85, 0.63], frameon=True)
    plt.rc('pdf', fonttype=42)  # IMPORTANT to get rid of Type 3

    plt.rcParams.update(params)
    matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    matplotlib.rc('text', usetex=True)

    colors, linestyles = get_colorful_styles()
    f, ax1 = plt.subplots()

    legend_custom = {
        'adv_success': 'Malicious objective',
        'test_accuracy': 'Benign objective'
    }
    linestyles_custom = {
        'adv_success': ':',
        'test_accuracy': '-'
    }
    colors_custom = {
        'resnet_cifar_greencars_lm_cmp_evaluation': colors[0],
        'e44_cifar_resnet_racing_stripes_evaluation': colors[1],
    }

    for id, type in enumerate(plot_types):
        for suffix in ['adv_success', 'test_accuracy']:
            # print(f"{type}/{suffix}")
            # print(df[f"{type}/{suffix}"])
            # df.plot(x='Round', y=plot_legend[type], style='o', label=plot_legend[type], color=colors[id], linestyle=linestyles[id], linewidth=2)
            plt.plot(df.Round, df[f"{type}/{suffix}"], label=legend_custom[suffix], color=colors_custom[type],
                     linestyle=linestyles_custom[suffix], linewidth=2)

    plt.xlabel('Round')
    # ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: '%.0fk' % (y * 1e-3)))

    plt.ylabel("Accuracy")
    plt.ylim(ymin=0, ymax=1.0)
    plt.xlim(xmin=-5, xmax=430)
    start, end = ax1.get_xlim()
    xticks = np.arange(0, end + 1, 100)
    # np.insert(xticks, 0, -5, axis=0)
    ax1.xaxis.set_ticks(xticks)

    # Additional, custom legend
    custom_lines_colors = [  # Line2D([0], [0], linestyle="-", lw=2, color=colors[2]),
        Line2D([0], [0], linestyle="-", lw=2, color=colors[1]),
        Line2D([0], [0], linestyle="-", lw=2, color=colors[0])]
    custom_lines_styles = [Line2D([0], [0], linestyle="-", lw=2, color=COLOR_GRAY),
                           Line2D([0], [0], linestyle=":", lw=2, color=COLOR_GRAY)]
    leg1 = plt.legend(custom_lines_colors, ["Model replacement", "Subspace poisoning"],
                      bbox_to_anchor=(1., 0.69, 1., .102), loc=3, ncol=1, columnspacing=0.75)
    leg2 = plt.legend(custom_lines_styles, ["Benign objective", "Malicious objective"],
                      bbox_to_anchor=(1., 0.39, 1., .102), loc=3, ncol=1, columnspacing=0.75)
    ax1.add_artist(leg1)
    ax1.add_artist(leg2)

    plt.grid(True, linestyle=':', color='0.8', zorder=0)
    F = plt.gcf()
    F.set_size_inches(fig_size)
    pdf_pages.savefig(F, bbox_inches='tight')
    plt.clf()
    pdf_pages.close()

def prio_accuracy_plot(plotname):
    data_prio = {
        5: 0.002185,
        25: 0.106695,
        250: 0.308112,
        2500: 2.59,
        32768: 34.884178,
        50000: 44.497297,
        100000: 89.693458,
        200000: 187.233027
    }
    data_me = {
        1024: 445.9 / 1000.0,
        2048: 897.2 / 1000.0,
        4096: 1798.075 / 1000.0,
        8192: 3601.8 / 1000.0,
        16384: 7221.125 / 1000.0,
        32768: 14621.25 / 1000.0
    }

    params, fig_size = get_plt_params()

    pdf_pages = PdfPages('./plots/%s' % plotname)

    plt.rcParams.update(params)
    plt.axes([0.12, 0.32, 0.85, 0.63], frameon=True)
    plt.rc('pdf', fonttype=42)  # IMPORTANT to get rid of Type 3

    plt.rcParams.update(params)
    matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    matplotlib.rc('text', usetex=True)

    colors, linestyles = get_colorful_styles()
    f, ax1 = plt.subplots()

    p_x, p_y = zip(*data_prio.items())
    m_x, m_y = zip(*data_me.items())
    plt.plot(p_x, p_y, '-o', color=colors[0], label="Prio", linewidth=2)
    plt.plot(m_x, m_y, '-o', color=colors[1], label="Bulletproofs", linewidth=2)

    plt.xlabel('Parameters')
    # ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: '%.0fk' % (y * 1e-3)))
    # plt.ylim(ymin, ymax)
    plt.ylabel("Time")
    plt.title("Range proof generation time per client")

    plt.legend()
    plt.grid(True, linestyle=':', color='0.8', zorder=0)
    F = plt.gcf()
    F.set_size_inches(fig_size)
    pdf_pages.savefig(F, bbox_inches='tight')
    plt.clf()
    pdf_pages.close()


def endtoend_accuracy_plot(plotname, dataset, title, ymin, ymax):
    plot_types = {
        f'{dataset}_plain_baseline.csv': "Plain",
        f'{dataset}_range_old_slow.csv': "Na\\\"{i}ve",
        f'{dataset}_range_optim_slow.csv': "Optimized"
    }
    eval_save_path = os.path.join(plot_data_save_path, "endtoend")

    params, fig_size = get_plt_params()

    pdf_pages = PdfPages('./plots/%s' % plotname)

    plt.rcParams.update(params)
    plt.axes([0.12, 0.32, 0.85, 0.63], frameon=True)
    plt.rc('pdf', fonttype=42)  # IMPORTANT to get rid of Type 3

    plt.rcParams.update(params)
    matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    matplotlib.rc('text', usetex=True)

    colors, linestyles = get_colorful_styles()
    f, ax1 = plt.subplots()

    for id, (type, name) in enumerate(plot_types.items()):
        df = pd.read_csv(os.path.join(eval_save_path, type), header=None)
        # print(type, df[0], df[2])
        plt.plot(df[0][:40], df[2][:40], '-o', color=colors[id], label=name, linewidth=2)

    # for id, type in enumerate(plot_types):
    #     for suffix in ['adv_success', 'test_accuracy']:
    #         # print(f"{type}/{suffix}")
    #         # print(df[f"{type}/{suffix}"])
    #         # df.plot(x='Round', y=plot_legend[type], style='o', label=plot_legend[type], color=colors[id], linestyle=linestyles[id], linewidth=2)
    #         plt.plot(df.Round, df[f"{type}/{suffix}"], label=legend_custom[suffix], color=colors_custom[type], linestyle=linestyles_custom[suffix], linewidth=2)

    plt.xlabel('Round')
    # ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: '%.0fk' % (y * 1e-3)))
    plt.ylim(ymin, ymax)
    plt.ylabel("Accuracy")
    plt.title(title)

    plt.legend()
    plt.grid(True, linestyle=':', color='0.8', zorder=0)
    F = plt.gcf()
    F.set_size_inches(fig_size)
    pdf_pages.savefig(F, bbox_inches='tight')
    plt.clf()
    pdf_pages.close()


def endtoend_accuracy_four_plot(plotname):
    plot_types = {
        "mnist": {
            "plain": "mnist_plain_baseline.csv",
            "range": {
                f'mnist_range_old_slow.csv': "Na\\\"{i}ve",
                f'mnist_range_optim_randproof.csv': "Optimized"
            },
            "l2": {
                f'mnist_range_old_slow.csv': "Na\\\"{i}ve",
                f'mnist_l2_optim.csv': "Optimized"  # TODO!
            }
        },
        "cifar": {
            "plain": "cifar_lenet_plain_baseline.csv",
            "range": {
                f'cifar_lenet_range_old_slow.csv': "Na\\\"{i}ve",
                f'cifar_lenet_range_optim_slow.csv': "Optimized"
            },
            "l2": {
                f'cifar_lenet_range_old_slow.csv': "Na\\\"{i}ve",
                f'cifar_lenet_l2_optim.csv': "Optimized"
            }
        }
    }
    eval_save_path = os.path.join(plot_data_save_path, "endtoend")

    params, fig_size = get_plt_params()

    _, fig_size, _ = get_large_figsize(450.0, 0.7)

    pdf_pages = PdfPages('./plots/%s' % plotname)

    plt.rcParams.update(params)
    # plt.axes([0.12, 0.32, 0.85, 0.63], frameon=True)
    plt.rc('pdf', fonttype=42)  # IMPORTANT to get rid of Type 3

    plt.rcParams.update(params)
    matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    matplotlib.rc('text', usetex=True)

    colors, linestyles = get_colorful_styles()
    f, axs = plt.subplots(2, 2)

    labels = {
        "range": "$L_\\infty$",
        "l2": "$L_2$",
        "mnist": "MNIST",
        "cifar": "CIFAR-10"
    }

    for id, (dataset, bounds) in enumerate(plot_types.items()):
        dfplain = pd.read_csv(os.path.join(eval_save_path, bounds["plain"]), header=None)
        for x in [0, 1]:
            axs[id, x].plot(dfplain[0][:40], dfplain[2][:40], '-o', color=colors[0], label="Plain", linewidth=2)
        axs[id, 0].set(ylabel=labels[dataset])

        for index, bound in enumerate(["range", "l2"]):
            axs[1, index].set(xlabel=labels[bound])
            for optimizedIndex, (filename, label) in enumerate(bounds[bound].items()):
                df = pd.read_csv(os.path.join(eval_save_path, filename), header=None)
                axs[id, index].plot(df[0][:40], df[2][:40], '-o', color=colors[optimizedIndex + 1], label=label,
                                    linewidth=2)

    for i in [0, 1]:
        axs[0, i].set_ylim(0.8, 1.0)
        axs[1, i].set_ylim(0, 0.6)

    for ax in axs.flat:
        ax.grid(True, linestyle=':', color='0.8', zorder=0)

        # df = pd.read_csv(os.path.join(eval_save_path, type), header=None)
        # # print(type, df[0], df[2])
        # plt.plot(df[0][:40], df[2][:40], '-o', color=colors[id], label=name, linewidth=2)

    # for id, type in enumerate(plot_types):
    #     for suffix in ['adv_success', 'test_accuracy']:
    #         # print(f"{type}/{suffix}")
    #         # print(df[f"{type}/{suffix}"])
    #         # df.plot(x='Round', y=plot_legend[type], style='o', label=plot_legend[type], color=colors[id], linestyle=linestyles[id], linewidth=2)
    #         plt.plot(df.Round, df[f"{type}/{suffix}"], label=legend_custom[suffix], color=colors_custom[type], linestyle=linestyles_custom[suffix], linewidth=2)

    # for ax in axs.flat:
    # ax.set(xlabel='Round', ylabel='Accuracy', ylim=(0.0, 1.0))
    # for ax in axs.flat:
    #     ax.label_outer()
    # plt.xlabel('Round')
    # ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: '%.0fk' % (y * 1e-3)))
    # plt.ylim(0.0, 1.0)
    # plt.ylabel("Accuracy")
    # plt.title("Title")

    plt.legend()
    plt.grid(True, linestyle=':', color='0.8', zorder=0)
    F = plt.gcf()
    F.set_size_inches(fig_size)
    pdf_pages.savefig(F, bbox_inches='tight')
    plt.clf()
    pdf_pages.close()

def bandwidth_bounds_four_plot(plotname):
    params, fig_size = get_plt_params()
    params['legend.fontsize'] = FONT_SIZE - 4

    _, fig_size, _ = get_large_figsize(450.0, 0.5)

    pdf_pages = PdfPages('./plots/%s' % plotname)

    plt.rcParams.update(params)
    # plt.axes([0.12, 0.32, 0.85, 0.63], frameon=True)
    plt.rc('pdf', fonttype=42)  # IMPORTANT to get rid of Type 3

    plt.rcParams.update(params)
    matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    matplotlib.rc('text', usetex=True)

    colors, linestyles = get_colorful_styles()
    f, axs = plt.subplots(1, 2)

    labels = {
        "range": "$L_\\infty$",
        "l2": "$L_2$",
        "mnist": "MNIST",
        "cifar": "CIFAR-10"
    }

    def next_pow(x):
        # print(x)
        return pow(2, math.ceil(math.log(x, 2)))

    def linf_baseline(D, n, p):
        return 32 * 2 * D,\
               32 * p * (math.log(n, 2) + math.log(next_pow(D / p)) + 9),\
               32 * 4 * D
        # return 32 * (6 * D + p * (math.log(n, 2) + math.log(next_pow(D / p)) + 9))

    def l2_baseline(D, n, p):
        return 32 * 2 * D, \
               32 * p * (math.log(n, 2) + math.log(next_pow(D / p)) + 9), \
               32 * 6 * D,\
                32 * D,\
                math.log(n, 2) + 9
        # return 32 * (9 * D + p * (math.log(n, 2) + math.log(next_pow(D / p)) + 9) + math.log(n, 2) + 9)

    def plaintext(D):
        return 4 * D
    n = 32
    p = 64

    print(linf_baseline(pow(2, 15), n, p))

    x = list(range(1, int(math.pow(2, 15)), 1000))
    # print([linf_baseline(y, n, p) for y in x])
    axs[0].stackplot(x, *zip(*[linf_baseline(y, n, p) for y in x]), linewidth=2, colors=colors)
    axs[1].stackplot(x, *zip(*[l2_baseline(y, n, p) for y in x]), linewidth=2, labels=["Commitments", "Range proofs", "Randomness proofs", "Squared commitments", "$L_2$-norm range proof"], colors=colors)

    mkfunc = lambda x, pos: '%1.1f' % (x * 1e-6) if x >= 1e6 else '%1.1fK' % (x * 1e-3) if x >= 1e3 else '%1.1f' % x
    mkformatter = matplotlib.ticker.FuncFormatter(mkfunc)

    axs[0].set(ylabel="Message size (Mbytes)")

    axs[0].set(xlabel="Parameters ($L_\\infty$)")
    axs[1].set(xlabel="Parameters ($L_2$)")
    for id in [0, 1]:
        axs[id].set_ylim(0, 10000000)

    axs[0].plot(x, [plaintext(y) for y in x], linewidth=2, color='#000000', linestyle='--')
    axs[1].plot(x, [plaintext(y) for y in x], linewidth=2, color='#000000', label="Plaintext", linestyle='--')

    axs[1].legend(bbox_to_anchor=(-.49, 1.), loc="upper right", ncol=1, columnspacing=0.75)
    # axs[1].legend(loc="upper left", prop=fontP)
    for ax in axs.flat:
        ax.yaxis.set_major_formatter(mkformatter)
    # axs[0, 0].plot(x, dfplain[2][:40], '-o', color=colors[0], label="Plain", linewidth=2)

    # for id, (dataset, bounds) in enumerate(plot_types.items()):
    #     dfplain = pd.read_csv(os.path.join(eval_save_path, bounds["plain"]), header=None)
    #     for x in [0, 1]:
    #         axs[id, x].plot(dfplain[0][:40], dfplain[2][:40], '-o', color=colors[0], label="Plain", linewidth=2)
    #     axs[id, 0].set(ylabel=labels[dataset])
    #
    #     for index, bound in enumerate(["range", "l2"]):
    #         axs[1, index].set(xlabel=labels[bound])
    #         for optimizedIndex, (filename, label) in enumerate(bounds[bound].items()):
    #             df = pd.read_csv(os.path.join(eval_save_path, filename), header=None)
    #             axs[id, index].plot(df[0][:40], df[2][:40], '-o', color=colors[optimizedIndex + 1], label=label,
    #                                 linewidth=2)
    #
    # for i in [0, 1]:
    #     axs[0, i].set_ylim(0.8, 1.0)
    #     axs[1, i].set_ylim(0, 0.6)

    for ax in axs.flat:
        ax.grid(True, linestyle=':', color='0.8', zorder=0)

    plt.grid(True, linestyle=':', color='0.8', zorder=0)
    F = plt.gcf()
    F.set_size_inches(fig_size)
    pdf_pages.savefig(F, bbox_inches='tight')
    plt.clf()
    pdf_pages.close()

def weight_distribution_plot_plus_l2(plotname):
    tags = [
        'histogram_ben/l6_Dense',
        'histogram_ben/l0_Conv2D',
        'histogram_ben/l2_Conv2D',
        'histogram_ben/l4_Dense',
        'histogram_ben/l8_Dense'
    ]
    tags_mal = [
        'histogram_mal/l6_Dense',
        'histogram_mal/l0_Conv2D',
        'histogram_mal/l2_Conv2D',
        'histogram_mal/l4_Dense',
        'histogram_mal/l8_Dense'
    ]
    print(matplotlib.font_manager.weight_dict)
    del matplotlib.font_manager.weight_dict['roman']
    matplotlib.font_manager._rebuild()
    print(matplotlib.font_manager.weight_dict)

    params, fig_size = get_plt_params()

    params['legend.fontsize'] = 14
    params['legend.title_fontsize'] = 16

    pdf_pages = PdfPages('./plots/%s' % plotname)

    # plt.rcParams.update(params)
    plt.axes([0.12, 0.32, 0.85, 0.63], frameon=True)
    plt.rc('pdf', fonttype=42)  # IMPORTANT to get rid of Type 3

    plt.rcParams.update(params)
    # matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    # matplotlib.rc('text', usetex=True)

    f, ax1 = plt.subplots()

    display_hist = True
    display_kde = False
    # How many weights outside bound
    bound = 0.01

    _, dist_mal = extract_histogram(
        './data/histograms/cifar_lenet_bgwall_40_dist.events',
        tags_mal,
        [5])  # For now
    print(dist_mal.shape)

    cnt = np.count_nonzero(dist_mal > bound) + np.count_nonzero(dist_mal < -bound)
    print("Malicious: ", cnt, cnt / dist_mal.shape[0])

    bins = np.arange(-0.1125, 0.1125, 0.005)
    sns.distplot(dist_mal, bins=bins, hist=display_hist, kde=display_kde, norm_hist=True,
                 kde_kws={'shade': True, 'linewidth': 0}, hist_kws={'alpha': 1}, color=colors[0], label="Malicious", ax=ax1)
    del dist_mal

    _, dist_ben_nineteen = extract_histogram(
        './data/histograms/cifar_lenet_bgwall_40_dist.events',
        tags,
        [5])

    # shuffle randomly, then select
    bins = np.arange(-0.03263, 0.02697, 0.0009)
    dist_ben = dist_ben_nineteen
    print(dist_ben.shape)



    cnt = np.count_nonzero(dist_ben_nineteen > bound) + np.count_nonzero(dist_ben_nineteen < -bound)
    print("Benign:", cnt, cnt / dist_ben.shape[0])

    sns.distplot(dist_ben, bins=bins, hist=display_hist, kde=display_kde, norm_hist=True,
                 kde_kws={'shade': True, 'linewidth': 0}, hist_kws={'alpha': 1, 'weights': np.repeat(1. / 19., dist_ben.shape[0])},
                 color=colors[2], label="Benign", ax=ax1)
    del dist_ben
    print('Done with ben')

    # print("Second attack")
    # _, dist_mal_modelreplacement = extract_histogram(
    #     '../../experiments_set/cifar_lenet/cifar_lenet_bgwall_40_mr_dist/events/events.out.tfevents.1592169131.ip-172-31-1-86.eu-central-1.compute.internal',
    #     tags_mal,
    #     [5])  # For now
    # print(dist_mal_modelreplacement.shape)
    # # bins = np.arange(-0.1125, 0.1125, 0.005)
    # sns.distplot(dist_mal_modelreplacement, hist=display_hist, kde=display_kde, norm_hist=True,
    #              kde_kws={'shade': True, 'linewidth': 0}, color=colors[2], label="Malicious (model replacement)", ax=ax1)

    # plt.hist(dist_mal)

    ax1.set_xlabel("Weight")
    ax1.set_ylabel("Density (KDE)")
    ax1.set_axisbelow(True)
    # plt.yscale("log")

    custom_benign = [Patch(facecolor=colors[2], label="Benign (0.99)"),
                     Patch(facecolor=colors[0], label="Malicious (8.38)")]
    leg1 = plt.legend(handles=custom_benign, title="Client type ($L_2$)")
    leg1._legend_box.align = "left"

    plt.grid(True, axis="y", linestyle=':', color='0.8', zorder=0, linewidth=1.2)
    F = plt.gcf()
    F.set_size_inches(fig_size)
    pdf_pages.savefig(F, bbox_inches='tight')
    plt.clf()
    pdf_pages.close()

def weight_distribution_plot_plus_l2_attack(plotname):
    tags = [
        'histogram_ben/l6_Dense',
        'histogram_ben/l0_Conv2D',
        'histogram_ben/l2_Conv2D',
        'histogram_ben/l4_Dense',
        'histogram_ben/l8_Dense'
    ]
    tags_mal = [
        'histogram_mal/l6_Dense',
        'histogram_mal/l0_Conv2D',
        'histogram_mal/l2_Conv2D',
        'histogram_mal/l4_Dense',
        'histogram_mal/l8_Dense'
    ]

    params, fig_size = get_plt_params()

    pdf_pages = PdfPages('./plots/%s' % plotname)

    # plt.rcParams.update(params)
    plt.axes([0.12, 0.32, 0.85, 0.63], frameon=True)
    plt.rc('pdf', fonttype=42)  # IMPORTANT to get rid of Type 3

    plt.rcParams.update(params)
    matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    matplotlib.rc('text', usetex=True)

    f, ax1 = plt.subplots()

    # _, dist_ben_nineteen = extract_histogram(
    #     '../../experiments_set/cifar_lenet/cifar_lenet_bgwall_40_dist/events/events.out.tfevents.1592158373.ip-172-31-1-86.eu-central-1.compute.internal',
    #     tags,
    #     [5])
    #
    # # shuffle randomly, then select
    # bins = np.arange(-0.03263, 0.02697, 0.0009)
    # dist_ben = dist_ben_nineteen
    # print(dist_ben.shape)
    display_hist = True
    display_kde = False
    # sns.distplot(dist_ben, bins=bins, hist=display_hist, kde=display_kde, norm_hist=True,
    #              kde_kws={'shade': True, 'linewidth': 0}, hist_kws={'weights': np.repeat(1. / 19., dist_ben.shape[0])},
    #              color=colors[0], label="Benign", ax=ax1)
    # del dist_ben
    print('Done with ben')
    _, dist_mal = extract_histogram(
        '../../experiments_set/cifar_lenet/cifar_lenet_bgwall_40_dist/events/events.out.tfevents.1592158373.ip-172-31-1-86.eu-central-1.compute.internal',
        tags_mal,
        [5])  # For now
    print(dist_mal.shape)

    bins = np.arange(-0.1125, 0.1125, 0.005)
    sns.distplot(dist_mal, bins=bins, hist=display_hist, kde=display_kde, norm_hist=True,
                 kde_kws={'shade': True, 'linewidth': 0}, color=colors[0], label="Malicious", ax=ax1)
    del dist_mal

    print("Second attack")
    scale_factor = 40. / 100.
    _, dist_mal_modelreplacement = extract_histogram(
        '../../experiments_set/cifar_lenet/cifar_lenet5_bgwall/run-3/events/events.out.tfevents.1591807122.ip-172-31-1-86.eu-central-1.compute.internal',
        tags_mal,
        [5])  # For now
    print(dist_mal_modelreplacement.shape)
    # bins = np.arange(-0.1125, 0.1125, 0.005)
    dist_mal_modelreplacement = dist_mal_modelreplacement * scale_factor
    sns.distplot(dist_mal_modelreplacement, hist=display_hist, kde=display_kde, norm_hist=True,
                 kde_kws={'shade': True, 'linewidth': 0}, color=colors[1], label="Malicious (model replacement)", ax=ax1)


    ax1.set_xlabel("Weight")
    ax1.set_ylabel("Density")
    # plt.yscale("log")

    custom_benign = [Patch(facecolor=colors[0], label="Subspace p. (8.38)"),
                     Patch(facecolor=colors[1], label="Model repl. (8.38)")]
    leg1 = plt.legend(handles=custom_benign, title="Client type ($L_2$)")
    leg1._legend_box.align = "left"

    plt.grid(True, linestyle=':', color='0.8', zorder=0)
    F = plt.gcf()
    F.set_size_inches(fig_size)
    pdf_pages.savefig(F, bbox_inches='tight')
    plt.clf()
    pdf_pages.close()

def quantization_mnist(plotname):
    df = pd.read_csv(os.path.join(plot_data_save_path, 'quantization_emnist.csv'))
    # print(df)
    plot_types = [
        # 'quantization_baseline_evaluation/test_accuracy',
        #           'quantization_prob_evaluation/test_accuracy',
        #           'quantization_deterministic_evaluation/test_accuracy',
        #           'quantization_prob_higher_loss_evaluation/test_accuracy'
        'quantization_emnist_baseline_evaluation/test_accuracy',
        'quantization_emnist_p_8_7_evaluation/test_accuracy',
        'quantization_emnist_p_4_3_evaluation/test_accuracy',
        'quantization_emnist_d_8_7_evaluation/test_accuracy',
        # 'quantization_mnist5_prob_1_1_evaluation/test_accuracy'
    ]
    plot_legend = {
        # 'quantization_baseline_evaluation/test_accuracy': "No quantization",
        #           'quantization_prob_evaluation/test_accuracy': "(16-7)-p)",
        #           'quantization_deterministic_evaluation/test_accuracy': "(16-7)-d",
        #            "quantization_prob_higher_loss_evaluation/test_accuracy": "(8-4)-p"

        'quantization_emnist_baseline_evaluation/test_accuracy': '32-bit float',
        'quantization_emnist_p_8_7_evaluation/test_accuracy': '(8,7)-prob.',
        'quantization_emnist_p_4_3_evaluation/test_accuracy': '(4,3)-prob.',
        'quantization_emnist_d_8_7_evaluation/test_accuracy': '(8,7)-det.',
        # 'quantization_mnist5_prob_1_1_evaluation/test_accuracy': "(1-1)-p"
    }

    fig_height, fig_size, fig_width = get_large_figsize()

    params, fig_size = get_plt_params()

    pdf_pages = PdfPages('./plots/%s' % plotname)

    plt.rcParams.update(params)
    plt.axes([0.12, 0.32, 0.85, 0.63], frameon=True)
    plt.rc('pdf', fonttype=42)  # IMPORTANT to get rid of Type 3
    matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    matplotlib.rc('text', usetex=True)

    colors, linestyles = get_colorful_styles()
    f, ax1 = plt.subplots()

    linestyles = ["-", "--", ":"]

    for id, type in enumerate(plot_types):
        # df.plot(x='Round', y=plot_legend[type], style='o', label=plot_legend[type], color=colors[id], linestyle=linestyles[id], linewidth=2)
        plt.plot(df.Round[0:1100], df[type][0:1100], label=plot_legend[type], color=colors[id], linestyle="-",
                 linewidth=2)

    plt.xlabel('Round')
    # ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: '%.0fk' % (y * 1e-3)))

    plt.ylabel("Accuracy")

    plt.ylim(0.9, 1.0)
    # plt.xlim(0, 1000)

    plt.legend()
    # plt.legend(bbox_to_anchor=(-0.016, 1., 1., .102), loc=3, ncol=4, columnspacing=0.75)

    plt.grid(True, linestyle=':', color='0.8', zorder=0)
    F = plt.gcf()
    F.set_size_inches(fig_size)
    pdf_pages.savefig(F, bbox_inches='tight')
    plt.clf()
    pdf_pages.close()


def cifar_client_comparison_unbounded(plotname):
    df = pd.read_csv(os.path.join(plot_data_save_path, 'cifar_lenet_client_comparison.csv'))
    # print(df)
    runs = {
        "run-0": 0.02, "run-1": 0.01, "run-2": 1. / 150., "run-3": 0.005
    }

    fig_height, fig_size, fig_width = get_large_figsize()

    params, fig_size = get_plt_params()

    pdf_pages = PdfPages('./plots/%s' % plotname)

    plt.rcParams.update(params)
    plt.axes([0.12, 0.32, 0.85, 0.63], frameon=True)
    plt.rc('pdf', fonttype=42)  # IMPORTANT to get rid of Type 3
    matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    matplotlib.rc('text', usetex=True)

    colors, linestyles = get_colorful_styles()
    f, ax1 = plt.subplots()

    values_tuples = [(alpha, df[f"{type}_evaluation/adv_success"][4]) for (type, alpha) in runs.items()]
    values = list(zip(*values_tuples))

    # for id, (type, alpha) in enumerate(runs.items()):
    #     # df.plot(x='Round', y=plot_legend[type], style='o', label=plot_legend[type], color=colors[id], linestyle=linestyles[id], linewidth=2)
    #     key = f"{type}_evaluation/adv_success"

    plt.plot(values[0], values[1], '-o', color=colors[1], label="Green cars", linestyle="-", linewidth=2)

    plt.xlabel('Adversarial fraction $\\alpha$')
    # ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: '%.0fk' % (y * 1e-3)))

    ax1.xaxis.set_major_formatter(ticker.PercentFormatter())
    plt.ylabel("Adversarial accuracy")
    plt.ylim(0.5, 1.0)
    # plt.xlim(0, 1000)

    plt.legend(loc='lower right')
    # plt.legend(bbox_to_anchor=(-0.016, 1., 1., .102), loc=3, ncol=4, columnspacing=0.75)

    plt.grid(True, linestyle=':', color='0.8', zorder=0)
    F = plt.gcf()
    F.set_size_inches(fig_size)
    pdf_pages.savefig(F, bbox_inches='tight')
    plt.clf()
    pdf_pages.close()


def modelreplacement_cifar_clip_plot(plotname):
    df = pd.read_csv(os.path.join(plot_data_save_path, 'modelreplacement.csv'))
    df["Round"] = df["Round"].apply(lambda x: x - 5)

    # print(df)

    def cust(plt, ax):
        plt.xlim(xmin=-5, xmax=300)
        plt.ylim(ymin=0, ymax=1.0)

        start, end = ax.get_xlim()
        xticks = np.arange(0, end + 1, 20)
        # np.insert(xticks, 0, -5, axis=0)
        ax.xaxis.set_ticks(xticks)

    plot_types = ['Adversarial objective (clipped)', 'Benign objective (clipped)']
    plot_legend = {'Benign objective (clipped)': 'Benign objective',
                   'Adversarial objective (clipped)': 'Adversarial objective'}
    plot_accuracy_round(plotname, df, plot_types, plot_legend, cust)


def constant_attack_lenet_bound_plot(plotname):
    df = pd.read_csv(os.path.join(plot_data_save_path, 'constant_attack_lenet_bound_plot.csv'))
    df["Round"] = df["Round"].apply(lambda x: x - 5)
    # print(df)
    plot_types = [
        'cifar_lenet_noniid_evaluation',
                  'cifar_lenet_train_noattack_clip_100_evaluation',
                  'cifar_lenet_train_repeated_greencar_100_evaluation'
                  ]

    params, fig_size = get_plt_params()

    pdf_pages = PdfPages('./plots/%s' % plotname)

    plt.rcParams.update(params)
    plt.axes([0.12, 0.32, 0.85, 0.63], frameon=True)
    plt.rc('pdf', fonttype=42)  # IMPORTANT to get rid of Type 3

    plt.rcParams.update(params)
    matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    matplotlib.rc('text', usetex=True)

    colors, linestyles = get_colorful_styles()
    f, ax1 = plt.subplots()

    legend_custom = {
        'adv_success': 'Malicious objective',
        'test_accuracy': 'Benign objective'
    }
    linestyles_custom = {
        'adv_success': ':',
        'test_accuracy': '-'
    }
    colors_custom = {
        'cifar_lenet_noniid_evaluation': colors[0],
        'cifar_lenet_train_noattack_clip_100_evaluation': colors[1],
        'cifar_lenet_train_repeated_greencar_100_evaluation': colors[2]  # add colors 1
    }

    for id, type in enumerate(plot_types[::-1]):
        for suffix in ['adv_success', 'test_accuracy']:
            # print(f"{type}/{suffix}")
            # print(df[f"{type}/{suffix}"])
            # df.plot(x='Round', y=plot_legend[type], style='o', label=plot_legend[type], color=colors[id], linestyle=linestyles[id], linewidth=2)
            plt.plot(df.Round, df[f"{type}/{suffix}"], label=legend_custom[suffix], color=colors_custom[type],
                     linestyle=linestyles_custom[suffix], linewidth=2)

    plt.xlabel('Round')
    # ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: '%.0fk' % (y * 1e-3)))

    plt.ylabel("Accuracy")
    plt.xlim(xmin=0, xmax=3927)
    start, end = ax1.get_xlim()
    # xticks = np.arange(0, end + 1, 100)
    # np.insert(xticks, 0, -5, axis=0)
    # ax1.xaxis.set_ticks(xticks)

    # Additional, custom legend
    custom_lines_colors = [Line2D([0], [0], linestyle="-", lw=2, color=colors[0]),
                           Line2D([0], [0], linestyle="-", lw=2, color=colors[1]),
                           Line2D([0], [0], linestyle="-", lw=2, color=colors[2])]
    custom_lines_styles = [Line2D([0], [0], linestyle="-", lw=2, color=COLOR_GRAY),
                           Line2D([0], [0], linestyle=":", lw=2, color=COLOR_GRAY)]
    leg1 = plt.legend(custom_lines_colors, ["Baseline", "Clipped ($L_2$)", "Attack, Clipped ($L_2$)"],
                      bbox_to_anchor=(1., 0.55, 1., .102), loc=3, ncol=1, columnspacing=0.75)
    leg2 = plt.legend(custom_lines_styles, ["Benign objective", "Malicious objective"],
                      bbox_to_anchor=(1., 0.26, 1., .102), loc=3, ncol=1,
                      columnspacing=0.75)
    ax1.add_artist(leg1)
    ax1.add_artist(leg2)

    plt.grid(True, linestyle=':', color='0.8', zorder=0)
    F = plt.gcf()
    F.set_size_inches(fig_size)
    pdf_pages.savefig(F, bbox_inches='tight')
    plt.clf()
    pdf_pages.close()


def plot_accuracy_round(plotname, df, plot_types, plot_legend, customize=None):
    fig_height, fig_size, fig_width = get_large_figsize()

    params, fig_size = get_plt_params()

    pdf_pages = PdfPages('./plots/%s' % plotname)

    plt.rcParams.update(params)
    plt.axes([0.12, 0.32, 0.85, 0.63], frameon=True)
    plt.rc('pdf', fonttype=42)  # IMPORTANT to get rid of Type 3
    matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    matplotlib.rc('text', usetex=True)

    colors, linestyles = get_colorful_styles()
    f, ax1 = plt.subplots()

    linestyles = ["-", "--", ":"]

    for id, type in enumerate(plot_types):
        # df.plot(x='Round', y=plot_legend[type], style='o', label=plot_legend[type], color=colors[id], linestyle=linestyles[id], linewidth=2)
        plt.plot(df.Round, df[type], label=plot_legend[type], color=colors[id], linestyle="-", linewidth=2)

    plt.xlabel('Round')
    # ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: '%.0fk' % (y * 1e-3)))

    plt.ylabel("Accuracy")

    if customize is not None:
        customize(plt, ax1)

    plt.legend(bbox_to_anchor=(-0.016, 1., 1., .102), loc=3, ncol=4, columnspacing=0.75)

    plt.grid(True, linestyle=':', color='0.8', zorder=0)
    F = plt.gcf()
    F.set_size_inches(fig_size)
    pdf_pages.savefig(F, bbox_inches='tight')
    plt.clf()
    pdf_pages.close()


if __name__ == "__main__":
    # e2e Plots
    selection = 'all'
    if len(sys.argv) > 1:
        selection = sys.argv[1]

    if selection == 'modelreplacement_cifar' or selection == 'all':
        modelreplacement_cifar_plot("modelreplacement_cifar.pdf")

    if selection == 'modelreplacement_subspacepoisoning_attack_compare' or selection == 'all':
        modelreplacement_subspacepoisoning_attack_compare("modelreplacement_subspacepoisoning_attack_compare.pdf")

    if selection == 'modelreplacement_cifar_clip' or selection == 'all':
        modelreplacement_cifar_clip_plot("modelreplacement_cifar_clip.pdf")
    if selection == 'inspectnorm_fmnist' or selection == 'all':
        inspect_norm_plot("inspectnorm_fmnist.pdf")
    if selection == 'inspectnorm_fmnist_lm_scale' or selection == 'all':
        inspect_norm_plot_lm_scale("inspectnorm_fmnist_lm_scale.pdf")
    if selection == 'microbenchmark_randproof' or selection == 'all':
        squarerandproof_log_plot("microbenchmark_create_randproof.pdf")

    if selection == 'microbenchmark_randproof' or selection == 'all':
        squarerandproof_verify_log_plot("microbenchmark_verify_randproof.pdf")

    if selection == 'norm_per_round' or selection == 'all':
        norm_per_round("norm_per_round.pdf")
    if selection == 'norm_distribution_benign' or selection == 'all':
        norm_distribution_benign("norm_distribution_benign.pdf")
    if selection == 'norm_distribution_benign_clients_multiround' or selection == 'all':
        norm_distribution_benign_clients_multiround("norm_distribution_benign_clients_multiround.pdf")
    if selection == 'norm_distribution_benign_overtime' or selection == 'all':
        norm_distribution_benign_overtime("norm_distribution_benign_overtime.pdf")

    if selection == 'constant_attack_lenet_bound_plot' or selection == 'all':
        constant_attack_lenet_bound_plot("constant_attack_lenet_bound_plot.pdf")

    if selection == 'l2_norm_accuracy_compare_plot' or selection == 'all':
        norm_accuracy_compare_plot("l2_norm_accuracy_compare_plot.pdf", "L2")
    if selection == 'linf_norm_accuracy_compare_plot' or selection == 'all':
        norm_accuracy_compare_plot("linf_norm_accuracy_compare_plot.pdf", "LINF")

    if selection == 'l2_norm_accuracy_tradeoff_fmnist_plot' or selection == 'all':
        norm_accuracy_tradeoff_plot("l2_norm_accuracy_tradeoff_fmnist_plot.pdf", "L2", DATA_KEYS_FMNIST, 'femnist_bounds_4.csv')
    if selection == 'linf_norm_accuracy_tradeoff_fmnist_plot' or selection == 'all':
        norm_accuracy_tradeoff_plot("linf_norm_accuracy_tradeoff_fmnist_plot.pdf", "LINF", DATA_KEYS_FMNIST, 'femnist_bounds_4.csv')
    if selection == 'l2_norm_accuracy_tradeoff_cifar_plot' or selection == 'all':
        norm_accuracy_tradeoff_plot("l2_norm_accuracy_tradeoff_cifar_plot.pdf", "L2", DATA_KEYS_CIFAR, 'cifar_bounds.csv')
    if selection == 'linf_norm_accuracy_tradeoff_cifar_plot' or selection == 'all':
        norm_accuracy_tradeoff_plot("linf_norm_accuracy_tradeoff_cifar_plot.pdf", "LINF", DATA_KEYS_CIFAR, 'cifar_bounds.csv')

    if selection == 'hypergeometric_distribution' or selection == 'all':
        hypergeometric_distribution("hypergeometric_distribution.pdf")

    if selection == 'quantization_mnist' or selection == 'all':
        quantization_mnist("quantization_mnist.pdf")

    if selection == 'l2proof_plots' or selection == 'all':
        l2proof_plots()
        l2proof_flexible_case()

    if selection == 'weight_distribution_plot_plus_l2' or selection == 'all':
        weight_distribution_plot_plus_l2("weight_distribution_plot_plus_l2.pdf")

    if selection == 'weight_distribution_plot_plus_l2_attack' or selection == 'all':
        weight_distribution_plot_plus_l2_attack("weight_distribution_plot_plus_l2_attack.pdf")

    if selection == 'cifar_client_comparison_unbounded' or selection == 'all':
        cifar_client_comparison_unbounded("cifar_client_comparison_unbounded.pdf")

    if selection == 'scaling_factor_adv_success' or selection == 'all':
        scaling_factor_adv_success("scaling_factor_adv_success.pdf")

    if selection == 'endtoend_mnist_cnn_range' or selection == 'all':
        endtoend_accuracy_plot("endtoend_mnist_cnn_range.pdf", "mnist", "$L_\\infty$-norm bound for the MNIST task.",
                               0.9, 1.0)

    if selection == 'endtoend_cifar_lenet_range' or selection == 'all':
        endtoend_accuracy_plot("endtoend_cifar_lenet_range.pdf", "cifar_lenet",
                               "$L_\\infty$-norm bound for the CIFAR10 task.", 0.0, 0.6)

    if selection == 'endtoend_accuracy_four_plot' or selection == 'all':
        endtoend_accuracy_four_plot("endtoend_accuracy_four_plot.pdf")

    if selection == 'endtoend_timing_bar_range' or selection == 'all':
        endtoend_timing_bar("endtoend_timing_bar_range.pdf", "range")

    if selection == 'endtoend_timing_bar_l2' or selection == 'all':
        endtoend_timing_bar("endtoend_timing_bar_l2.pdf", "l2")

    if selection == 'microbench_proof_arbitrary_ranges' or selection == 'all':
        microbench_proof_arbitrary_ranges()

    if selection == 'bandwidth_bounds_four_plot' or selection == 'all':
        bandwidth_bounds_four_plot("bandwidth_bounds_four_plot.pdf")

    if selection == 'accuracy_pgd' or selection == 'all':
        accuracy_pgd("accuracy_pgd.pdf")

    if selection == 'prio_accuracy_plot' or selection == 'all':
        prio_accuracy_plot("prio_accuracy_plot.pdf")

    if selection == 'cifar_lenet_wr_plot' or selection == 'all':
        cifar_lenet_wr_plot("cifar_lenet_wr_plot.pdf")