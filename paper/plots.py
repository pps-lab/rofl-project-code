#!/usr/bin/python
# coding=utf-8
import math
import sys
import os
import re

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
import matplotlib.patches as mpatches    

#from plotting.report.extract_histogram import extract_histogram
from extract_histogram import extract_histogram



plot_data_save_path = "./data/"
plots = "./images/"

COLOR_GRAY = "#AAAAAA"

FONT_SIZE = 20

DATA_KEYS = {
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
                'e41_clipinf_0.15_evaluation': 0.15
            },
            "PGD_ATTACK": {
            },
            "NO_ATTACK": {
            }
        }
    }
}


# Theming !
#output_dir = "."

def setup_plt(square=False):

    fig_width_pt = 240.0  # Get this from LaTeX using \showthe
    inches_per_pt = 1.0 / 72.27 * 2  # Convert pt to inches
    golden_mean = ((np.math.sqrt(5) - 1.0) / 2.0) * .8  # Aesthetic ratio
    fig_width = fig_width_pt * inches_per_pt  # width in inches
    fig_height = (fig_width * golden_mean)  # height in inches
    fig_size =  [fig_width, fig_height]

    if square:
        fig_size = [fig_height, fig_height]

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

    plt.rcParams.update(plt_params)
    plt.rc('pdf', fonttype=42)  # IMPORTANT to get rid of Type 3


def get_task_styling():
    task = {
            # Attacks
            "a2" : {
                "label": "A2-WALL",
                "color": "0.1"
            },
            "a3" : {
                "label": "A3-GREEN",
                "color": "0.3"
            },
            "a4": {
                "label": "A4-STRIPES",
                "color": "0.6"
            },

            # Metrics
            "main": { # accuracy
                "label": "Main Task",
                "linestyle": "dashdot"
            },
            "bdoor": { # accuracy
                "label": "Backdoor Task",
                "linestyle": "solid"  
            },
            "norm": {
                "label": "Norm",
                "linestyle": "dashdot"
            },


            # clients

            "benign_client": {
                "color": "black",
                "label": "Benign clients"
            }
    }
    return task



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


colors, linestyles = get_colorful_styles()


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

def norm_accuracy_tradeoff_plot(plotname, norm, output_dir, xtickspacing=None, xmax=None, add_legend=True, model="mnist"):
    df = pd.read_csv(os.path.join(plot_data_save_path, 'femnist_bounds_4.csv'))

    #df = pd.read_csv(os.path.join(plot_data_save_path, 'cifar_bounds.csv'))

    def build_df(df, norm, window_size, selected_round, pattern, col_baseline="e41_google_tasks_noconstrain_evaluation/test_accuracy", ignored_cols = ["e41_clipinf_0_03_evaluation/adv_success","e41_clipinf_0_03_evaluation/test_accuracy"]):

        lst = []
        used = []
        notused = []

        df["baseline_mean"] = df[col_baseline].rolling(window_size).mean()
        df["baseline_std"] = df[col_baseline].rolling(window_size).std()
        df_baseline = df[df["Round"]==selected_round]
        df_baseline = df_baseline[["Round", "baseline_mean", "baseline_std"]]
        df_baseline = df_baseline.rename(columns={"Round": "round"})

        bounds = {}


        for col in df.columns:

            match = re.search(pattern, col, re.IGNORECASE)
            if match:
                if col in ignored_cols:
                    print(f"Skipped (ignored): {col}")
                    notused.append(col)
                    continue


                try:
                    bound = float(match.group(2).replace("_", "."))
                except ValueError:
                    print(f"Skipped: {col}")
                    notused.append(col)
                    continue

                col_type = match.group(3)

                if f"{bound}_{col_type}" in bounds:
                    print(f"Skipped (Duplicate Bound): {col}")
                    notused.append(col)
                    continue
                else:
                    bounds[f"{bound}_{col_type}"] = True


                
                if col_type not in ["adv_success", "test_accuracy"]:
                    raise ValueError(f"Unknown col type: {col_type}")

                df[col + "_rmean"] = df[col].rolling(window_size).mean()
                df[col + "_rstd"] = df[col].rolling(window_size).std()

                row = df[df["Round"]==selected_round]

                d = {
                    "round": row["Round"].values[0],
                    "norm": norm,
                    "bound": bound,
                    col_type + "_mean": row[col + "_rmean"].values[0],
                    col_type + "_std": row[col + "_rstd"].values[0],
                }
                lst.append(d)
                used.append(col)

            else:
                notused.append(col)

        #print(f"Norm={norm}  - Ignored Columns: {notused}")

        df1 = pd.DataFrame(lst)

        # group together test accuracy and adv success
        df1 = df1.fillna(0)    
        df1 = df1.groupby(["round", "norm", "bound"]).agg({"test_accuracy_mean":"sum", "test_accuracy_std":"sum", "adv_success_mean": "sum", "adv_success_std": "sum"})
        # remove hierarchical index
        df1 = pd.DataFrame(df1.to_records())

        df1 = df1.merge(df_baseline)
        return df1



    setup_plt(square=False)
    name = plotname

    if norm == "l2" and model == "mnist":
        norm_label = "$L_2$"
        df = build_df(df, norm="l2", window_size=20, selected_round=670, pattern="e41_(emnist_)?clipl2_([0-9_\.]+)_evaluation/(.*)", col_baseline="e41_google_tasks_noconstrain_evaluation/test_accuracy", ignored_cols = ["e41_clipinf_0_03_evaluation/adv_success","e41_clipinf_0_03_evaluation/test_accuracy"])
        df = df[df["bound"]<100]
    elif norm == "l8" and model == "mnist":
        norm_label = "$L_{\infty}$"
        df = build_df(df, norm="l8", window_size=20, selected_round=670, pattern="e41_(emnist_)?clipinf_([0-9_\.]+)_evaluation/(.*)", col_baseline="e41_google_tasks_noconstrain_evaluation/test_accuracy", ignored_cols = ["e41_clipinf_0_03_evaluation/adv_success","e41_clipinf_0_03_evaluation/test_accuracy"])
        df = df[df["bound"]<=0.075]

    
    else: raise ValueError("unknown norm")

    colors = ["0.1", "0.3", "0.6"]
    ecolor=None #"0.6"
    linestyles = ["solid", "dotted"] #dashdot

    with PdfPages(f"{output_dir}/{name}.pdf") as pdf:
   
        fig, ax = plt.subplots()

        ##########################
        # Draw all the lines         
        ##########################

    
        baseline= ax.plot(df["bound"], df["baseline_mean"], label="Baseline (no bound)", color=colors[0],
             linestyle='dashdot', linewidth=2, alpha=0.5)

        testacc =  ax.errorbar(df["bound"], df["test_accuracy_mean"], yerr=df["test_accuracy_std"], label="Main Task", color=colors[0], linewidth=2, capsize=5, ecolor=ecolor, marker="o")
        advsucc = ax.errorbar(df["bound"], df["adv_success_mean"], yerr=df["adv_success_std"], label="Backdoor Task", color=colors[1], linestyle="dashed", linewidth=2, capsize=5, ecolor=ecolor, marker="o")


        ##########################
        # General Format         
        ##########################
        #ax.set_title("Hello World")
          # 'best', 'upper right', 'upper left', 'lower left', 
                                # 'lower right', 'right', 'center left',  'center right', 
                                # 'lower center', 'upper center', 'center'
        ax.grid(True, axis="y", linestyle=':', color='0.6', zorder=0, linewidth=1.2)

        if add_legend:
            ax.legend(title_fontsize=20, bbox_to_anchor=(0., 1.02, 2/3, .102), mode="expand", loc="lower left", title="Tasks", labelspacing=.05)  
            

        ##########################
        # Y - Axis Format
        ##########################
        ax.set_ylim(ymin=0, ymax=1.02)
        ax.set_ylabel("Accuracy")
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        #ax.set_yticklabels(labels, fontsize=16, rotation=345)


        ##########################
        # X - Axis Format
        ##########################
        ax.set_xlim(xmin=0, xmax=xmax)
        ax.set_xlabel(f"{norm_label} norm bound")

        import matplotlib.ticker as ticker
        ax.xaxis.set_major_locator(ticker.MultipleLocator(xtickspacing))
        #ax.set_xticks(xticks)
        #ax.set_xticklabels(labels, fontsize=16, rotation=345)

        if add_legend:
            ax.axis('off')

            baseline[0].set_visible(False)
            testacc[0].set_visible(False)
            testacc[1][0].set_visible(False)
            testacc[1][1].set_visible(False)
            testacc[2][0].set_visible(False)

            advsucc[0].set_visible(False)
            advsucc[1][0].set_visible(False)
            advsucc[1][1].set_visible(False)
            advsucc[2][0].set_visible(False)        

        pdf.savefig(bbox_inches='tight', pad_inches=0)
        plt.close()
    return fig, df




def get_plt_params():
    fig_height, fig_size, fig_width = get_large_figsize()

    params = {'backend': 'ps',
              'axes.labelsize': FONT_SIZE,
              'legend.fontsize': FONT_SIZE,
              'xtick.labelsize': FONT_SIZE,
              'ytick.labelsize': FONT_SIZE,
              'font.size': FONT_SIZE,
              'figure.figsize': fig_size,
              'font.family': 'times new roman'}
    return params, [fig_width, fig_height]


def norm_accuracy_compare_plot(plotname, norm, output_dir, legend_type=None, use_error=True, model="mnist", xmax=600, ignore_error=[], markevery=50):
    
    
    if legend_type not in [None, "tootight", "ideal", "tooloose"]:
        raise ValueError(f"legend type not supported: {legend_type}")

    window_size = 20

    if model == "mnist":

        df = pd.read_csv(os.path.join(plot_data_save_path, 'femnist_bounds_4.csv'))

        l2_bound_tootight = "e41_clipl2_0_01_evaluation"
        l2_bound_ideal = "e41_clipl2_1_evaluation"
        l2_bound_tooloose = "e41_clipl2_35_evaluation" #e41_clipl2_100_evaluation

        l8_bound_tootight = "e41_clipinf_0_0001_evaluation"
        l8_bound_ideal = "e41_clipinf_0_00100_evaluation"
        l8_bound_tooloose = "e41_emnist_clipinf_0_075_evaluation"

        tootight_bound = (r"10^{-2}", r"10^{-4}") #(L2, L8)
        ideal_bound = ("1", r"10^{-3}") #(L2, L8)
        tooloose_bound = ("35", "0.075") #(L2, L8)

    elif model == "cifar":
        df = pd.read_csv(os.path.join(plot_data_save_path, 'cifar_bounds.csv'))

        l2_bound_tootight = "e58_lr1_cifar_clipl2_0.5_evaluation"
        l2_bound_ideal = "e58_lr1_cifar_clipl2_10_evaluation"
        l2_bound_tooloose = "e58_lr1_cifar_baseline_evaluation"

        l8_bound_tootight = "e58_lr1_cifar_clip_0.004_evaluation"
        l8_bound_ideal = "e58_lr1_cifar_clip_0.0055_evaluation"
        l8_bound_tooloose = "e58_lr1_cifar_baseline_evaluation" 

        tootight_bound = ("0.5", "0.004") #(L2, L8)
        ideal_bound = ("10", "0.0055") #(L2, L8)
        tooloose_bound = ("\infty", "\infty") #(L2, L8)



    def build_df(df, norm, bound_tootight_key, bound_ideal_key, bound_tooloose_key, window_size):
        if bound_tootight_key is not None:
            df[f"{norm}_bound_tootight_advsuccess"] = df[f"{bound_tootight_key}/adv_success"].rolling(window_size).mean()
            df[f"{norm}_bound_tootight_testaccuracy"] = df[f"{bound_tootight_key}/test_accuracy"].rolling(window_size).mean()
            df[f"{norm}_bound_tootight_advsuccess_std"] = df[f"{bound_tootight_key}/adv_success"].rolling(window_size).std()
            df[f"{norm}_bound_tootight_testaccuracy_std"] = df[f"{bound_tootight_key}/test_accuracy"].rolling(window_size).std()
     
        if bound_ideal_key is not None:
            df[f"{norm}_bound_ideal_advsuccess"] = df[f"{bound_ideal_key}/adv_success"].rolling(window_size).mean()
            df[f"{norm}_bound_ideal_testaccuracy"] = df[f"{bound_ideal_key}/test_accuracy"].rolling(window_size).mean()
            df[f"{norm}_bound_ideal_advsuccess_std"] = df[f"{bound_ideal_key}/adv_success"].rolling(window_size).std()
            df[f"{norm}_bound_ideal_testaccuracy_std"] = df[f"{bound_ideal_key}/test_accuracy"].rolling(window_size).std()
        
        if bound_tooloose_key is not None:
            df[f"{norm}_bound_tooloose_advsuccess"] = df[f"{bound_tooloose_key}/adv_success"].rolling(window_size).mean()
            df[f"{norm}_bound_tooloose_testaccuracy"] = df[f"{bound_tooloose_key}/test_accuracy"].rolling(window_size).mean()
            df[f"{norm}_bound_tooloose_advsuccess_std"] = df[f"{bound_tooloose_key}/adv_success"].rolling(window_size).std()
            df[f"{norm}_bound_tooloose_testaccuracy_std"] = df[f"{bound_tooloose_key}/test_accuracy"].rolling(window_size).std()
        
        return df

    
    df = build_df(df, norm="l8", bound_tootight_key=l8_bound_tootight, bound_ideal_key=l8_bound_ideal, bound_tooloose_key=l8_bound_tooloose, window_size=window_size)
    df = build_df(df, norm="l2", bound_tootight_key=l2_bound_tootight, bound_ideal_key=l2_bound_ideal, bound_tooloose_key=l2_bound_tooloose, window_size=window_size)


    name = plotname
    setup_plt(square=False)
    

    with PdfPages(f"{output_dir}/{name}.pdf") as pdf:
   
        fig, ax = plt.subplots()

        ##########################
        # Draw all the lines         
        ##########################
        error_color = "0.85"
        colors = ["0.1", "0.3", "0.6"]
        linestyles = ["solid", "dotted"] #dashdot

        line_d = {}
        plines = []
        if f"{norm}_bound_tootight_testaccuracy" in df.columns:
            plines += ax.plot(df["Round"], df[f"{norm}_bound_tootight_testaccuracy"], color=colors[0], linestyle=linestyles[0], linewidth=2, marker="s", markevery=markevery)
            line_d["tootight_tacc"] = len(plines)-1
        if f"{norm}_bound_ideal_testaccuracy" in df.columns:
            plines += ax.plot(df["Round"], df[f"{norm}_bound_ideal_testaccuracy"], color=colors[1], linestyle=linestyles[0], linewidth=2, marker="o", markevery=markevery)
            line_d["ideal_tacc"] = len(plines)-1
        if f"{norm}_bound_tooloose_testaccuracy" in df.columns:
            plines += ax.plot(df["Round"], df[f"{norm}_bound_tooloose_testaccuracy"], color=colors[2], linestyle=linestyles[0], linewidth=2, marker="v", markevery=markevery)
            line_d["tooloose_tacc"] = len(plines)-1
        if f"{norm}_bound_tootight_advsuccess" in df.columns:
            plines += ax.plot(df["Round"], df[f"{norm}_bound_tootight_advsuccess"], color=colors[0], linestyle=linestyles[1], linewidth=2, marker="s", markevery=markevery)
            line_d["tootight_advs"] = len(plines)-1
        if f"{norm}_bound_ideal_advsuccess" in df.columns:
            plines += ax.plot(df["Round"], df[f"{norm}_bound_ideal_advsuccess"], color=colors[1], linestyle=linestyles[1], linewidth=2, marker="o", markevery=markevery)
            line_d["ideal_advs"] = len(plines)-1
        if f"{norm}_bound_tooloose_advsuccess" in df.columns:
            plines += ax.plot(df["Round"], df[f"{norm}_bound_tooloose_advsuccess"], color=colors[2], linestyle=linestyles[1], linewidth=2, marker="v", markevery=markevery)
            line_d["tooloose_advs"] = len(plines)-1

        lines = ax.get_lines()

        labels = ["Main Task", "Backdoor Task"]
        empty_patch = mpatches.Patch(color='none')

        handles=None
    
        if legend_type == "tootight" and "tootight_tacc" in line_d:
            title = "Bound too tight"
            labels = [f"($L_2 \leq {tootight_bound[0]}$, $L_{{\infty}} \leq {tootight_bound[1]}$)"] + labels
            handles = [empty_patch, lines[line_d["tootight_tacc"]], lines[line_d["tootight_advs"]]]
        elif legend_type == "ideal" and "ideal_tacc" in line_d:
            title = "Bound ideal"
            labels = [f"($L_2 \leq {ideal_bound[0]}$, $L_{{\infty}} \leq {ideal_bound[1]}$)"] + labels
            handles = [empty_patch, lines[line_d["ideal_tacc"]], lines[line_d["ideal_advs"]]]
        elif legend_type == "tooloose" and "tooloose_tacc" in line_d:
            title = "Bound too loose"
            labels = [f"($L_2 \leq {tooloose_bound[0]}$, $L_{{\infty}} \leq {tooloose_bound[1]}$)"] + labels
            handles = [empty_patch, lines[line_d["tooloose_tacc"]], lines[line_d["tooloose_advs"]]]
 
        if legend_type is not None and handles is not None:
            ax.legend(handles, labels, title_fontsize=20, bbox_to_anchor=(0., 1.02, 2/3, .102), mode="expand", loc="lower left", title=title, labelspacing=.05)  
            

        if use_error:
            
            if f"{norm}_bound_tootight_advsuccess" in df.columns:
                ax.fill_between(df["Round"], 
                        df[f"{norm}_bound_tootight_advsuccess"]-df[f"{norm}_bound_tootight_advsuccess_std"],
                        df[f"{norm}_bound_tootight_advsuccess"]+df[f"{norm}_bound_tootight_advsuccess_std"],
                        alpha=1, edgecolor='#3F7F4C', facecolor=error_color, linewidth=0)
            
            if f"{norm}_bound_tooloose_advsuccess" in df.columns and f"{norm}_bound_tooloose_advsuccess" not in ignore_error:
                ax.fill_between(df["Round"], 
                        df[f"{norm}_bound_tooloose_advsuccess"]-df[f"{norm}_bound_tooloose_advsuccess_std"],
                        df[f"{norm}_bound_tooloose_advsuccess"]+df[f"{norm}_bound_tooloose_advsuccess_std"],
                        alpha=1, edgecolor='#3F7F4C', facecolor=error_color, linewidth=0)

            elif f"{norm}_bound_tooloose_advsuccess" in ignore_error:
                ax.annotate('* std large', xy=(500, 0.32), color=colors[2], xycoords='data', xytext=(0, 0), textcoords='offset points', horizontalalignment='right', verticalalignment='bottom')


            if f"{norm}_bound_ideal_advsuccess" in df.columns:
                ax.fill_between(df["Round"], 
                        df[f"{norm}_bound_ideal_advsuccess"]-df[f"{norm}_bound_ideal_advsuccess_std"],
                        df[f"{norm}_bound_ideal_advsuccess"]+df[f"{norm}_bound_ideal_advsuccess_std"],
                        alpha=1, edgecolor='#3F7F4C', facecolor=error_color, linewidth=0)
                
            if f"{norm}_bound_tootight_testaccuracy" in df.columns:
                ax.fill_between(df["Round"], 
                        df[f"{norm}_bound_tootight_testaccuracy"]-df[f"{norm}_bound_tootight_testaccuracy_std"],
                        df[f"{norm}_bound_tootight_testaccuracy"]+df[f"{norm}_bound_tootight_testaccuracy_std"],
                        alpha=1, edgecolor='#3F7F4C', facecolor=error_color, linewidth=0)
            
            if f"{norm}_bound_tooloose_testaccuracy" in df.columns:
                ax.fill_between(df["Round"], 
                        df[f"{norm}_bound_tooloose_testaccuracy"]-df[f"{norm}_bound_tooloose_testaccuracy_std"],
                        df[f"{norm}_bound_tooloose_testaccuracy"]+df[f"{norm}_bound_tooloose_testaccuracy_std"],
                        alpha=1, edgecolor='#3F7F4C', facecolor=error_color, linewidth=0)
            
            if f"{norm}_bound_ideal_testaccuracy" in df.columns:
                ax.fill_between(df["Round"], 
                        df[f"{norm}_bound_ideal_testaccuracy"]-df[f"{norm}_bound_ideal_testaccuracy_std"],
                        df[f"{norm}_bound_ideal_testaccuracy"]+df[f"{norm}_bound_ideal_testaccuracy_std"],
                        alpha=1, edgecolor='#3F7F4C', facecolor=error_color, linewidth=0)



        ##########################
        # General Format         
        ##########################
        #ax.set_title("Hello World")

        ax.grid(True, axis="y", linestyle=':', color='0.6', zorder=0, linewidth=1.2)


        ##########################
        # Y - Axis Format
        ##########################
        ax.set_ylim(ymin=0, ymax=1.01)
        ax.set_ylabel("Accuracy")
        ax.set_yticks([0,0.25, 0.5, 0.75, 1])
        #ax.set_yticklabels(labels, fontsize=16, rotation=345)


        ##########################
        # X - Axis Format
        ##########################

        ax.set_xlim(xmin=0, xmax=xmax)

        ax.set_xlabel("Rounds")

        #ax.set_xticks(xticks)
        #ax.set_xticklabels(labels, fontsize=16, rotation=345)

        if legend_type is not None:
            ax.axis('off')
            for line in plines:
                line.set_visible(False)


        pdf.savefig(bbox_inches='tight', pad_inches=0)
        plt.close()
    return fig, df
    
    
    

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


def build_df_scaling_norm_advsuccess(prefix):
    SCALING_FACTORS = {
        10: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], # 10 clients selected -> have 10 scaling factors
        20: [1, 3, 5, 7, 9, 11, 13, 15, 17, 19], # 20 client selected -> have 10 different scaling factors
        40: [1, 5, 9, 13, 17, 23, 27, 31, 35, 40] # 40 clients -> have 10 different scaling factors
        # 40: [1, 10, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 90, 100]
    }

    folder = "./data/l2_comparison_attack"

    df_10 = pd.DataFrame(SCALING_FACTORS[10], columns=["scaling_factor"]) 
    df_10["n_clients"] = 10

    df_20 = pd.DataFrame(SCALING_FACTORS[20], columns=["scaling_factor"]) 
    df_20["n_clients"] = 20

    df_40 = pd.DataFrame(SCALING_FACTORS[40], columns=["scaling_factor"]) 
    df_40["n_clients"] = 40


    task_translation = {
        "bgwall": "a2-wall",
        "greencar": "a3-green",
        "racingstripes": "a4-stripes"
    }

    for filename in os.listdir(folder):
        pattern = f"{prefix}_([a-z]+)_([0-9]+).csv"
        match = re.search(pattern, filename, re.IGNORECASE)
        if match:
            attack_task = match.group(1)
            n_clients = int(match.group(2))


            df1 = pd.read_csv(f"{folder}/{filename}")
            df1 = df1.tail(n=1) # attack happens only in last round (round 5)

            # select and sort all backdoor columns and all norm columns
            advsucc_cols = [col for col in df1.columns if "/adv_success" in col]
            l2norm_cols = [col for col in df1.columns if "_l2_total/mal"in col]

            advsucc_cols.sort()
            l2norm_cols.sort()

            # extract two columns and merge them into df
            df_advsucc = pd.DataFrame(df1[advsucc_cols].transpose().values, columns=[f"{task_translation[attack_task]}_bdoor"])
            df_l2norm = pd.DataFrame(df1[l2norm_cols].transpose().values, columns=[f"{task_translation[attack_task]}_l2norm"])

            df_cc = pd.concat([df_advsucc, df_l2norm], axis=1)
            df_sorted = df_cc.sort_values(f"{task_translation[attack_task]}_l2norm").reset_index(drop=True)

            if n_clients == 10:
                df_10 = pd.concat([df_10, df_sorted], axis=1)
            elif n_clients == 20:
                df_20 = pd.concat([df_20, df_sorted], axis=1)
            elif n_clients == 40:
                df_40  = pd.concat([df_40, df_sorted], axis=1)
            else:
                print(f"Ignore file: {filename} with n_clients={n_clients}")


        else:
            print(f"no match: {filename}")


    df = pd.concat([df_10, df_20, df_40])


    df["alpha_fracadv"] = 1 / df["n_clients"]
    return df




def scaling_factor_adv_success(plotname, output_dir, prefix=None, df=None):

    if prefix is not None:
        df = build_df_scaling_norm_advsuccess(prefix)

    df = df[df["n_clients"]==40]
    

    setup_plt()
    task = get_task_styling()
    name = plotname

    with PdfPages(f"{output_dir}/{name}.pdf") as pdf:
   
        fig, ax = plt.subplots()
        ax2 = ax.twinx()
        ##########################
        # Draw all the lines         
        ##########################

        linewidth = 1.5
        ax2.plot(df["scaling_factor"], df["a2-wall_bdoor"], marker="o", color=task["a2"]["color"], linestyle=task["bdoor"]["linestyle"], linewidth=linewidth)
        ax2.plot(df["scaling_factor"], df["a3-green_bdoor"], marker="o", color=task["a3"]["color"], linestyle=task["bdoor"]["linestyle"], linewidth=linewidth)
        ax2.plot(df["scaling_factor"], df["a4-stripes_bdoor"], marker="o", color=task["a4"]["color"], linestyle=task["bdoor"]["linestyle"], linewidth=linewidth)

        ax.plot(df["scaling_factor"], df["a2-wall_l2norm"], color=task["a2"]["color"], linestyle=task["norm"]["linestyle"], linewidth=linewidth)
        ax.plot(df["scaling_factor"], df["a3-green_l2norm"], color=task["a3"]["color"], linestyle=task["norm"]["linestyle"], linewidth=linewidth)
        ax.plot(df["scaling_factor"], df["a4-stripes_l2norm"], color=task["a4"]["color"], linestyle=task["norm"]["linestyle"], linewidth=linewidth)

        ##########################
        # General Format         
        ##########################

        ax.grid(True, axis="y", linestyle=':', color='0.6', zorder=0, linewidth=1.2)

        ## Additional, custom legend                   
        patches = [mpatches.Patch(color=task["a2"]["color"]), mpatches.Patch(color=task["a3"]["color"]), mpatches.Patch(color=task["a4"]["color"])]
        
        custom_lines_styles = [Line2D([0], [0], linestyle=task["norm"]["linestyle"], lw=2, color=COLOR_GRAY),
                               Line2D([0], [0], linestyle=task["bdoor"]["linestyle"], lw=2, color=COLOR_GRAY)]
        
        height = 0
        width = 0.48
        leg1 = ax.legend(patches, [task["a2"]["label"], task["a3"]["label"], task["a4"]["label"]],
                          mode="expand", title="Attack Tasks", bbox_to_anchor=(1.15, 1, width, height), loc="upper left", labelspacing=0.2)
        
        leg2 = ax.legend(custom_lines_styles, [task["norm"]["label"], task["bdoor"]["label"]],
                          mode="expand", title="Metrics", bbox_to_anchor=(1.15, 0, width, height), loc="lower left", labelspacing=0.2)
        ax.add_artist(leg1)
        ax.add_artist(leg2)

        ##########################
        # Y - Axis Format
        ##########################
        ax.set_ylim(ymin=0, ymax=None)
        ax.set_ylabel("$L_2$ Norm of Update")

        ax2.set_ylim(ymin=0, ymax=1.02)
        ax2.set_ylabel("Task Accuracy")
        ax2.set_yticks([0, 0.25, 0.5, 0.75, 1])
        #ax.set_yticklabels(labels, fontsize=16, rotation=345)

        ##########################
        # X - Axis Format
        ##########################
        ax.set_xlim(xmin=0, xmax=None)
        ax.set_xlabel("Scaling factor")
        #ax.set_xticks(xticks)
        #ax.set_xticklabels(labels, fontsize=16, rotation=345)

        pdf.savefig(bbox_inches='tight', pad_inches=0)
        plt.close()
    return fig, df



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


def norm_distribution_benign(plotname, output_dir):

    df = build_df_scaling_norm_advsuccess("cifar_lenet_minloss_wr")

    name = plotname
    setup_plt()

    task = get_task_styling()

    with PdfPages(f"{output_dir}/{name}.pdf") as pdf:
   
        fig, ax = plt.subplots()

        ##########################
        # Draw all the lines         
        ##########################

        for i in [6]:
            file = np.load(f'./data/cifar_lenet/noniid_norms/round_{i}.npy',
                           allow_pickle=True)
            # file = np.load(f'../../experiments_set/norm/normround/round_{i}.npy', allow_pickle=True)
            benign_norms_l2, benign_norms_l1, mal_norms_l2, mal_norms_l1 = file[0], file[1], file[2], file[3]
            sns.distplot(benign_norms_l2, hist=False, kde=True, color="black", norm_hist=True,

                         kde_kws={'shade': True, 'linewidth': 2, "alpha":0, "hatch": "///"}, ax=ax)

        ax2 = ax.twinx()

        alphas = {
            0.025: {
                "label": "2.5 %",
                "linestyle": "dashed"
            },
            0.05: {
                "label": "5 %",
                "linestyle": "dashdot"
            },

            0.1:{
                "label": "10 %",
                "linestyle": "solid"
            }
        }

        for alpha in df["alpha_fracadv"].unique():
            df1 = df[df["alpha_fracadv"]==alpha]

            df1.sort_values("a2-wall_l2norm", inplace=True)
            ax2.plot(df1["a2-wall_l2norm"], df1["a2-wall_bdoor"], linestyle=alphas[alpha]["linestyle"], marker="o", color=task["a2"]["color"])
            
            df1.sort_values("a3-green_l2norm", inplace=True)
            ax2.plot(df1["a3-green_l2norm"], df1["a3-green_bdoor"], linestyle=alphas[alpha]["linestyle"], marker="o", color=task["a3"]["color"])

            df1.sort_values("a4-stripes_l2norm", inplace=True)
            ax2.plot(df1["a4-stripes_l2norm"], df1["a4-stripes_bdoor"], linestyle=alphas[alpha]["linestyle"], marker="o", color=task["a4"]["color"])

        

        ##########################
        # General Format         
        ##########################

        ax.grid(True, axis="y", linestyle=':', color='0.6', zorder=0, linewidth=1.2)



        ## Additional, custom legend                   
        patches = [mpatches.Patch(color=task["a2"]["color"]), mpatches.Patch(color=task["a3"]["color"]), mpatches.Patch(color=task["a4"]["color"])]
        
        


        matplotlib.rcParams['hatch.linewidth'] = 2
        custom_lines_styles = [Line2D([0], [0], linestyle=alphas[0.025]["linestyle"], lw=2, color=COLOR_GRAY),
                               Line2D([0], [0], linestyle=alphas[0.05]["linestyle"], lw=2, color=COLOR_GRAY),
                               Line2D([0], [0], linestyle=alphas[0.1]["linestyle"], lw=2, color=COLOR_GRAY)]
        
        height = 0
        width = 0.48


        leg0 = ax.legend([mpatches.Patch(facecolor="white" , edgecolor="black", hatch="///", linewidth=2)], [task["benign_client"]["label"]], loc="lower right")


        leg1 = ax.legend(patches, [task["a2"]["label"], task["a3"]["label"], task["a4"]["label"]],
                          mode="expand", title="Attack Tasks", bbox_to_anchor=(1.15, 1.05, width, height), loc="upper left", labelspacing=0.2)
        
        leg2 = ax.legend(custom_lines_styles, [alphas[0.025]["label"], alphas[0.05]["label"], alphas[0.1]["label"]],
                          mode="expand", title=r"$\alpha$ (attackers)", bbox_to_anchor=(1.15, -0.05, width, height), loc="lower left", labelspacing=0.2)
        ax.add_artist(leg0)
        ax.add_artist(leg1)
        ax.add_artist(leg2)


        ##########################
        # Y - Axis Format
        ##########################

        ax.set_ylabel("Density (KDE)")

        ax2.set_ylim(ymin=0, ymax=1.02)
        ax2.set_ylabel("Backdoor Accuracy")
        ax2.set_yticks([0, 0.25, 0.5, 0.75, 1])
        #ax.set_yticklabels(labels, fontsize=16, rotation=345)


        ##########################
        # X - Axis Format
        ##########################
        ax.set_xlim(xmin=0, xmax=None)
        ax.set_xlabel("$L_2$ Norm of Updates")
        #ax.set_xticks(yticks)
        #ax.set_xticklabels(labels, fontsize=16, rotation=345)

        pdf.savefig(bbox_inches='tight', pad_inches=0)
        plt.close()
    return fig, df













def norm_distribution_iid_noniid(plotname):
    pdf_pages = PdfPages('./plots/%s' % plotname)
    params, fig_size = get_plt_params()

    plt.rcParams.update(params)
    matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    matplotlib.rc('text', usetex=True)

    plt.axes([0.12, 0.32, 0.85, 0.63], frameon=True)
    plt.rc('pdf', fonttype=42)  # IMPORTANT to get rid of Type 3

    colors, linestyles = get_colorful_styles()
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
        file = np.load(f'../../experiments_set/cifar_lenet/dist_noniid/norms/round_{round_num}.npy',
                   allow_pickle=True)
        benign_norms_l2.extend(file[0])
        benign_norms_l1.extend(file[1])
    # file = np.load(f'../../experiments_set/norm/normround/round_{i}.npy', allow_pickle=True)
    # benign_norms_l2, benign_norms_l1 = file[0], file[1]
    print("NonIID", benign_norms_l2)
    sns.distplot(benign_norms_l2, hist=False, kde=True,
                 kde_kws={'shade': True, 'linewidth': 0}, ax=ax1)
    plt.axvline(x=1.9, ymin=0, ymax=1, label="Norm bound (1.9)", linestyle="--", color=colors[1])

    ax1.set_xlabel('$L_2$-norm')
    # ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: '%.0fk' % (y * 1e-3)))

    ax1.set_ylabel("Percentage of benign users")
    # plt.xscale("log")
    # plt.yscale("log")
    plt.legend(bbox_to_anchor=(-0.016, 1.00, 1., .102), loc=3, ncol=4, columnspacing=0.75)

    plt.grid(True, linestyle=':', color='0.8', zorder=0)
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

# TODO [nku] adjust color scheme
def modelreplacement_cifar_resnet56_plot(plotname, output_dir):
    df = pd.read_csv(os.path.join(plot_data_save_path, 'e44_cifar_resnet.csv'))

    # NEW

    df1 = df[["Round"]]
    df1 = df1.rename(columns={"Round": "round"})

    # rename cols
    for suffix, short in [("test_accuracy", "testacc"), ("adv_success", "advsucc")]:
        
        df1[f"a2-wall_{short}"] = df[f"e44_cifar_attack_400_0.0001_full_evaluation/{suffix}"]
        df1[f"a3-green_{short}"] = df[f"e44_cifar_attack_400_0.0001_full_greencars_evaluation/{suffix}"]
        df1[f"a4-stripes_{short}"] = df[f"e44_cifar_resnet_racing_stripes_evaluation/{suffix}"]

    
    df = df1


    task = get_task_styling()

    name = plotname
    setup_plt()
    
    with PdfPages(f"{output_dir}/{name}.pdf") as pdf:
   
        fig, ax = plt.subplots()

        ##########################
        # Draw all the lines         
        ##########################
        linewidth = 1.5
        ax.plot(df["round"], df[f"a2-wall_testacc"], color=task["a2"]["color"], linestyle=task["main"]["linestyle"], linewidth=linewidth)
        ax.plot(df["round"], df[f"a3-green_testacc"], color=task["a3"]["color"], linestyle=task["main"]["linestyle"], linewidth=linewidth)
        ax.plot(df["round"], df[f"a4-stripes_testacc"], color=task["a4"]["color"], linestyle=task["main"]["linestyle"], linewidth=linewidth)

        ax.plot(df["round"], df[f"a2-wall_advsucc"], color=task["a2"]["color"], linestyle=task["bdoor"]["linestyle"], linewidth=linewidth)
        ax.plot(df["round"], df[f"a3-green_advsucc"], color=task["a3"]["color"], linestyle=task["bdoor"]["linestyle"], linewidth=linewidth)
        ax.plot(df["round"], df[f"a4-stripes_advsucc"], color=task["a4"]["color"], linestyle=task["bdoor"]["linestyle"], linewidth=linewidth)


        ##########################
        # General Format         
        ##########################
            
        ## Additional, custom legend                   
        patches = [mpatches.Patch(color=task["a2"]["color"]), mpatches.Patch(color=task["a3"]["color"]), mpatches.Patch(color=task["a4"]["color"])]
        

        custom_lines_styles = [Line2D([0], [0], linestyle=task["main"]["linestyle"], lw=2, color=COLOR_GRAY),
                               Line2D([0], [0], linestyle=task["bdoor"]["linestyle"], lw=2, color=COLOR_GRAY)]
        
        height = 0
        width = 0.48
        leg1 = ax.legend(patches, [task["a2"]["label"], task["a3"]["label"], task["a4"]["label"]],
                          mode="expand", title="Attack Tasks", bbox_to_anchor=(1, 1, width, height), loc="upper left", labelspacing=0.2)

        
        leg2 = ax.legend(custom_lines_styles, [task["main"]["label"], task["bdoor"]["label"]],
                          mode="expand", title="Metrics", bbox_to_anchor=(1, 0, width, height), loc="lower left", labelspacing=0.2)
        ax.add_artist(leg1)
        ax.add_artist(leg2)
        
        ax.grid(True, axis="y", linestyle=':', color='0.6', zorder=0, linewidth=1.2)


        ##########################
        # Y - Axis Format
        ##########################
        ax.set_ylim(ymin=0, ymax=1.02)
        ax.set_ylabel("Task Accuracy")
        ax.set_yticks([0,0.25, 0.5, 0.75, 1])
        #ax.set_yticklabels(labels, fontsize=16, rotation=345)


        ##########################
        # X - Axis Format
        ##########################
        ax.set_xlim(xmin=0, xmax=300)
        ax.set_xlabel("Rounds")
        #ax.set_xticks(xticks)
        #ax.set_xticklabels(labels, fontsize=16, rotation=345)

        pdf.savefig(bbox_inches='tight', pad_inches=0)
        plt.close()

    return fig, df

def modelreplacement_cifar_resnet18_plot(plotname, output_dir):
    df = pd.read_csv(os.path.join(plot_data_save_path, 'e44_cifar_resnet18.csv'))

    # NEW

    df1 = df[["Round"]]
    df1 = df1.rename(columns={"Round": "round"})

    # rename cols
    for suffix, short in [("test_accuracy", "testacc"), ("adv_success", "advsucc")]:

        df1[f"a2-wall_{short}"] = df[f"e3_cifar_resnet18_long_WALL_lrlow10_evaluation/{suffix}"]
        df1[f"a3-green_{short}"] = df[f"e3_cifar_resnet18_long_GREEN_lrlow10_evaluation/{suffix}"]
        df1[f"a4-stripes_{short}"] = df[f"e3_cifar_resnet18_long_STRIPES_lrlow10_evaluation/{suffix}"]


    df = df1


    task = get_task_styling()

    name = plotname
    setup_plt()

    with PdfPages(f"{output_dir}/{name}.pdf") as pdf:

        fig, ax = plt.subplots()

        ##########################
        # Draw all the lines
        ##########################
        linewidth = 1.5
        ax.plot(df["round"], df[f"a2-wall_testacc"], color=task["a2"]["color"], linestyle=task["main"]["linestyle"], linewidth=linewidth)
        ax.plot(df["round"], df[f"a3-green_testacc"], color=task["a3"]["color"], linestyle=task["main"]["linestyle"], linewidth=linewidth)
        ax.plot(df["round"], df[f"a4-stripes_testacc"], color=task["a4"]["color"], linestyle=task["main"]["linestyle"], linewidth=linewidth)

        ax.plot(df["round"], df[f"a2-wall_advsucc"], color=task["a2"]["color"], linestyle=task["bdoor"]["linestyle"], linewidth=linewidth)
        ax.plot(df["round"], df[f"a3-green_advsucc"], color=task["a3"]["color"], linestyle=task["bdoor"]["linestyle"], linewidth=linewidth)
        ax.plot(df["round"], df[f"a4-stripes_advsucc"], color=task["a4"]["color"], linestyle=task["bdoor"]["linestyle"], linewidth=linewidth)


        ##########################
        # General Format
        ##########################

        ## Additional, custom legend
        patches = [mpatches.Patch(color=task["a2"]["color"]), mpatches.Patch(color=task["a3"]["color"]), mpatches.Patch(color=task["a4"]["color"])]


        custom_lines_styles = [Line2D([0], [0], linestyle=task["main"]["linestyle"], lw=2, color=COLOR_GRAY),
                               Line2D([0], [0], linestyle=task["bdoor"]["linestyle"], lw=2, color=COLOR_GRAY)]

        height = 0
        width = 0.48
        leg1 = ax.legend(patches, [task["a2"]["label"], task["a3"]["label"], task["a4"]["label"]],
                         mode="expand", title="Attack Tasks", bbox_to_anchor=(1, 1, width, height), loc="upper left", labelspacing=0.2)


        leg2 = ax.legend(custom_lines_styles, [task["main"]["label"], task["bdoor"]["label"]],
                         mode="expand", title="Metrics", bbox_to_anchor=(1, 0, width, height), loc="lower left", labelspacing=0.2)
        ax.add_artist(leg1)
        ax.add_artist(leg2)

        ax.grid(True, axis="y", linestyle=':', color='0.6', zorder=0, linewidth=1.2)


        ##########################
        # Y - Axis Format
        ##########################
        ax.set_ylim(ymin=0, ymax=1.02)
        ax.set_ylabel("Task Accuracy")
        ax.set_yticks([0,0.25, 0.5, 0.75, 1])
        #ax.set_yticklabels(labels, fontsize=16, rotation=345)


        ##########################
        # X - Axis Format
        ##########################
        ax.set_xlim(xmin=0, xmax=20)
        ax.set_xlabel("Rounds")
        #ax.set_xticks(xticks)
        #ax.set_xticklabels(labels, fontsize=16, rotation=345)

        pdf.savefig(bbox_inches='tight', pad_inches=0)
        plt.close()

    return fig, df

def modelreplacement_cifar_resnet18_lowerlr_plot(plotname, output_dir):
    df = pd.read_csv(os.path.join(plot_data_save_path, 'e44_cifar_resnet18.csv'))

    # NEW

    df1 = df[["Round"]]
    df1 = df1.rename(columns={"Round": "round"})

    # rename cols
    for suffix, short in [("test_accuracy", "testacc"), ("adv_success", "advsucc")]:

        df1[f"a2-wall_{short}"] = df[f"e3_cifar_resnet18_long_WALL_lrlow_evaluation/{suffix}"]
        df1[f"a3-green_{short}"] = df[f"e3_cifar_resnet18_long_GREEN_lrlow100_evaluation/{suffix}"]
        df1[f"a4-stripes_{short}"] = df[f"e3_cifar_resnet18_long_STRIPES_lrlow100_evaluation/{suffix}"]


    df = df1


    task = get_task_styling()

    name = plotname
    setup_plt()

    with PdfPages(f"{output_dir}/{name}.pdf") as pdf:

        fig, ax = plt.subplots()

        ##########################
        # Draw all the lines
        ##########################
        linewidth = 1.5
        ax.plot(df["round"], df[f"a2-wall_testacc"], color=task["a2"]["color"], linestyle=task["main"]["linestyle"], linewidth=linewidth)
        ax.plot(df["round"], df[f"a3-green_testacc"], color=task["a3"]["color"], linestyle=task["main"]["linestyle"], linewidth=linewidth)
        ax.plot(df["round"], df[f"a4-stripes_testacc"], color=task["a4"]["color"], linestyle=task["main"]["linestyle"], linewidth=linewidth)

        ax.plot(df["round"], df[f"a2-wall_advsucc"], color=task["a2"]["color"], linestyle=task["bdoor"]["linestyle"], linewidth=linewidth)
        ax.plot(df["round"], df[f"a3-green_advsucc"], color=task["a3"]["color"], linestyle=task["bdoor"]["linestyle"], linewidth=linewidth)
        ax.plot(df["round"], df[f"a4-stripes_advsucc"], color=task["a4"]["color"], linestyle=task["bdoor"]["linestyle"], linewidth=linewidth)


        ##########################
        # General Format
        ##########################

        ## Additional, custom legend
        patches = [mpatches.Patch(color=task["a2"]["color"]), mpatches.Patch(color=task["a3"]["color"]), mpatches.Patch(color=task["a4"]["color"])]


        custom_lines_styles = [Line2D([0], [0], linestyle=task["main"]["linestyle"], lw=2, color=COLOR_GRAY),
                               Line2D([0], [0], linestyle=task["bdoor"]["linestyle"], lw=2, color=COLOR_GRAY)]

        height = 0
        width = 0.48
        leg1 = ax.legend(patches, [task["a2"]["label"], task["a3"]["label"], task["a4"]["label"]],
                         mode="expand", title="Attack Tasks", bbox_to_anchor=(1, 1, width, height), loc="upper left", labelspacing=0.2)


        leg2 = ax.legend(custom_lines_styles, [task["main"]["label"], task["bdoor"]["label"]],
                         mode="expand", title="Metrics", bbox_to_anchor=(1, 0, width, height), loc="lower left", labelspacing=0.2)
        ax.add_artist(leg1)
        ax.add_artist(leg2)

        ax.grid(True, axis="y", linestyle=':', color='0.6', zorder=0, linewidth=1.2)


        ##########################
        # Y - Axis Format
        ##########################
        ax.set_ylim(ymin=0, ymax=1.02)
        ax.set_ylabel("Task Accuracy")
        ax.set_yticks([0,0.25, 0.5, 0.75, 1])
        #ax.set_yticklabels(labels, fontsize=16, rotation=345)


        ##########################
        # X - Axis Format
        ##########################
        ax.set_xlim(xmin=0, xmax=300)
        ax.set_xlabel("Rounds")
        #ax.set_xticks(xticks)
        #ax.set_xticklabels(labels, fontsize=16, rotation=345)

        pdf.savefig(bbox_inches='tight', pad_inches=0)
        plt.close()

    return fig, df


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

    params, fig_size = get_plt_params()

    pdf_pages = PdfPages('./plots/%s' % plotname)

    # plt.rcParams.update(params)
    plt.axes([0.12, 0.32, 0.85, 0.63], frameon=True)
    plt.rc('pdf', fonttype=42)  # IMPORTANT to get rid of Type 3

    plt.rcParams.update(params)
    matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    matplotlib.rc('text', usetex=True)

    f, ax1 = plt.subplots()

    _, dist_ben_nineteen = extract_histogram(
        '../../experiments_set/cifar_lenet/cifar_lenet_bgwall_40_dist/events/events.out.tfevents.1592158373.ip-172-31-1-86.eu-central-1.compute.internal',
        tags,
        [5])

    # shuffle randomly, then select
    bins = np.arange(-0.03263, 0.02697, 0.0009)
    dist_ben = dist_ben_nineteen
    print(dist_ben.shape)
    display_hist = True
    display_kde = False
    sns.distplot(dist_ben, bins=bins, hist=display_hist, kde=display_kde, norm_hist=True,
                 kde_kws={'shade': True, 'linewidth': 0}, hist_kws={'weights': np.repeat(1. / 19., dist_ben.shape[0])},
                 color=colors[0], label="Benign", ax=ax1)
    del dist_ben
    print('Done with ben')
    _, dist_mal = extract_histogram(
        '../../experiments_set/cifar_lenet/cifar_lenet_bgwall_40_dist/events/events.out.tfevents.1592158373.ip-172-31-1-86.eu-central-1.compute.internal',
        tags_mal,
        [5])  # For now
    print(dist_mal.shape)

    bins = np.arange(-0.1125, 0.1125, 0.005)
    sns.distplot(dist_mal, bins=bins, hist=display_hist, kde=display_kde, norm_hist=True,
                 kde_kws={'shade': True, 'linewidth': 0}, color=colors[1], label="Malicious", ax=ax1)
    del dist_mal

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
    ax1.set_ylabel("Density")
    # plt.yscale("log")

    custom_benign = [Patch(facecolor=colors[0], label="Benign (0.99)"),
                     Patch(facecolor=colors[1], label="Malicious (8.38)")]
    leg1 = plt.legend(handles=custom_benign, title="Client type ($L_2$)")
    leg1._legend_box.align = "left"

    plt.grid(True, linestyle=':', color='0.8', zorder=0)
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


def main():
    return
    # e2e Plots
    selection = 'all'
    if len(sys.argv) > 1:
        selection = sys.argv[1]

    if selection == 'modelreplacement_cifar' or selection == 'all':
        modelreplacement_cifar_resnet56_plot("modelreplacement_cifar.pdf")

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
    if selection == 'norm_distribution_iid_noniid' or selection == 'all':
        norm_distribution_iid_noniid("norm_distribution_iid_noniid.pdf")
    if selection == 'norm_distribution_benign_overtime' or selection == 'all':
        norm_distribution_benign_overtime("norm_distribution_benign_overtime.pdf")

    if selection == 'constant_attack_lenet_bound_plot' or selection == 'all':
        constant_attack_lenet_bound_plot("constant_attack_lenet_bound_plot.pdf")

    # TODO [nku] adjust to new style
    if selection == 'l2_norm_accuracy_compare_plot' or selection == 'all':
        norm_accuracy_compare_plot("l2_norm_accuracy_compare_plot.pdf", "L2")
    if selection == 'linf_norm_accuracy_compare_plot' or selection == 'all':
        norm_accuracy_compare_plot("linf_norm_accuracy_compare_plot.pdf", "LINF")

    if selection == 'l2_norm_accuracy_tradeoff_plot' or selection == 'all':
        norm_accuracy_tradeoff_plot("l2_norm_accuracy_tradeoff_plot.pdf", "L2")
    if selection == 'linf_norm_accuracy_tradeoff_plot' or selection == 'all':
        norm_accuracy_tradeoff_plot("linf_norm_accuracy_tradeoff_plot.pdf", "LINF")

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

    if selection == 'scaling_factor_adv_success_lenet' or selection == 'all':
        scaling_factor_adv_success("scaling_factor_adv_success_lenet.pdf")

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

if __name__ == "__main__":
    main()