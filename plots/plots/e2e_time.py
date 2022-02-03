
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from common import get_colorful_styles, output_dir


def setup_plt():

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
        'font.family': 'Times New Roman',
        'font.weight': 'normal'
    }

    plt.rcParams.update(plt_params)
    plt.rc('pdf', fonttype=42)  # IMPORTANT to get rid of Type 3


def build_fig_e2e_mnist_time(df, name="e2e_mnist_time"):

    colors, _ = get_colorful_styles()

    configs = {
        "u": {"label": "plain",                 "color": (0.44, 0.44, 0.44, 1), "linestyle": "--", "marker": "x"},
        "l8": {"label": "$L_{\infty}$",         "color": colors[1], "linestyle": "-", "marker": "x"},
        "l8p": {"label": "$L_{\infty}^{(p)}$",  "color": colors[1], "linestyle": "--", "marker": "x"},
        "l2": {"label": "$L_2$",                "color": colors[2], "linestyle": "-", "marker": "x"},
        "l2rst": {"label": "$L_2^{(rsl)}$",      "color": colors[2], "linestyle": "--", "marker": "x"}
    }
    setup_plt()

    with PdfPages(f"{output_dir}/{name}.pdf") as pdf:

        fig, ax = plt.subplots()

        ##########################
        # Draw all the lines
        ##########################

        # lineplot
        for suffix in configs.keys():
            values = df[f"accuracy_{suffix}"]
            labels = df[f"cumtime_{suffix}"]
            config = configs[suffix]
            plt.plot(labels, values, label=config["label"], color=config["color"], linestyle=config["linestyle"], marker=config["marker"], linewidth=2, markersize=8)


        ##########################
        # General Format
        ##########################
        # ax.set_title("Hello World")
        ax.legend(loc="best")   # 'best', 'upper right', 'upper left', 'lower left',
        # 'lower right', 'right', 'center left',  'center right',
        # 'lower center', 'upper center', 'center'
        ax.grid(True, axis="y", linestyle=':', color='0.6', zorder=0, linewidth=1.2)


        ##########################
        # Y - Axis Format
        ##########################
        ax.set_ylim(ymin=0, ymax=None)
        ax.set_ylabel("Accuracy")
        ax.set_yticks([0 ,0.25 ,0.5 ,0.75 ,1])
        # ax.set_yticklabels(labels, fontsize=16, rotation=345)


        ##########################
        # X - Axis Format
        ##########################
        # ax.set_xlim(xmin=-30, xmax=1000)
        ax.set_xlabel("Time [s] (log)")
        ax.set_xscale("log")
        # ax.set_xticks(xticks)
        # ax.set_xticklabels(labels, fontsize=16, rotation=345)

        pdf.savefig(bbox_inches='tight', pad_inches=0)
        plt.close()
    return fig


def build_fig_e2e_cifar_time(df, name="e2e_cifar_time"):

    colors, _ = get_colorful_styles()

    configs = {
        "u": {"label": "plain",                 "color": (0.44, 0.44, 0.44, 1), "linestyle": "--", "marker": "x"},
        "l8p": {"label": "$L_{\infty}^{(p)}$",  "color": colors[1], "linestyle": "--", "marker": "x"}
    }
    setup_plt()

    with PdfPages(f"{output_dir}/{name}.pdf") as pdf:

        fig, ax = plt.subplots()

        ##########################
        # Draw all the lines
        ##########################

        # lineplot
        for suffix in configs.keys():
            values = df[f"accuracy_{suffix}"]
            labels = df[f"cumtime_{suffix}"]
            config = configs[suffix]
            plt.plot(labels, values, label=config["label"], color=config["color"], linestyle=config["linestyle"], marker=config["marker"], linewidth=2, markersize=8)


        ##########################
        # General Format
        ##########################
        # ax.set_title("Hello World")
        ax.legend(loc="upper left")   # 'best', 'upper right', 'upper left', 'lower left',
        # 'lower right', 'right', 'center left',  'center right',
        # 'lower center', 'upper center', 'center'
        ax.grid(True, axis="y", linestyle=':', color='0.6', zorder=0, linewidth=1.2)


        ##########################
        # Y - Axis Format
        ##########################
        ax.set_ylim(ymin=0, ymax=None)
        ax.set_ylabel("Accuracy")
        ax.set_yticks([0 ,0.25 ,0.5 ,0.75 ,1])
        # ax.set_yticklabels(labels, fontsize=16, rotation=345)


        ##########################
        # X - Axis Format
        ##########################
        # ax.set_xlim(xmin=-30, xmax=1000)
        ax.set_xlabel("Time [s] (log)")
        ax.set_xscale("log")
        # ax.set_xticks(xticks)
        # ax.set_xticklabels(labels, fontsize=16, rotation=345)

        pdf.savefig(bbox_inches='tight', pad_inches=0)
        plt.close()
    return fig


def build_fig_e2e_shakespeare_time(df, name="e2e_shakespeare_time"):

    colors, _ = get_colorful_styles()

    configs = {
        "u": {"label": "plain",                 "color": (0.44, 0.44, 0.44, 1), "linestyle": "--", "marker": "x"},
        "l8p": {"label": "$L_{\infty}^{(p)}$",  "color": colors[1], "linestyle": "--", "marker": "x"}
    }
    setup_plt()

    with PdfPages(f"{output_dir}/{name}.pdf") as pdf:

        fig, ax = plt.subplots()

        ##########################
        # Draw all the lines
        ##########################

        # lineplot
        for suffix in configs.keys():
            values = df[f"accuracy_{suffix}"]
            labels = df[f"cumtime_{suffix}"]
            config = configs[suffix]
            plt.plot(labels, values, label=config["label"], color=config["color"], linestyle=config["linestyle"], marker=config["marker"], linewidth=2, markersize=8)


        ##########################
        # General Format
        ##########################
        # ax.set_title("Hello World")
        ax.legend(loc="upper left")   # 'best', 'upper right', 'upper left', 'lower left',
        # 'lower right', 'right', 'center left',  'center right',
        # 'lower center', 'upper center', 'center'
        ax.grid(True, axis="y", linestyle=':', color='0.6', zorder=0, linewidth=1.2)


        ##########################
        # Y - Axis Format
        ##########################
        ax.set_ylim(ymin=0, ymax=None)
        ax.set_ylabel("Accuracy")
        ax.set_yticks([0 ,0.25 ,0.5 ,0.75 ,1])
        # ax.set_yticklabels(labels, fontsize=16, rotation=345)


        ##########################
        # X - Axis Format
        ##########################
        # ax.set_xlim(xmin=-30, xmax=1000)
        ax.set_xlabel("Time [s] (log)")
        ax.set_xscale("log")
        # ax.set_xticks(xticks)
        # ax.set_xticklabels(labels, fontsize=16, rotation=345)

        pdf.savefig(bbox_inches='tight', pad_inches=0)
        plt.close()
    return fig
