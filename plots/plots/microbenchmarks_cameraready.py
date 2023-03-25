
# build large36core
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
from common import get_colorful_styles, output_dir


#pd.set_option('display.max_columns', 500)
#pd.set_option('display.max_rows', 500)

def setup_plt():

    fig_width_pt = 240  # Get this from LaTeX using \showthe
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

    plt.rcParams.update(plt_params)
    plt.rc('pdf', fonttype=42)  # IMPORTANT to get rid of Type 3

colors, _ = get_colorful_styles()

cagg_config = {
    "server_label": "aggregate commitment",
    "color": (0.44, 0.44, 0.44, 1)
}
wellformed_config = {
    "client_label": "create well-formedness proof",
    "server_label": "verify well-formedness proof",
    "color": colors[1]
}
range_config = {
    "client_label": "create range proof",
    "server_label": "verify range proof",
    "color": colors[2]
}



def extract(filename, pattern, label, data_dir, fixed_point_repr=None):
    # pattern group(1) represents fixed_point_repr
    # pattern group(2) represents n_weights

    lst = []
    match = re.search(pattern, filename, re.IGNORECASE)
    if match:
        # extract the weight parameter and the fp-repr
        group1 = int(match.group(1))

        if fixed_point_repr is None:
            fixed_point_repr = group1
            n_weights = int(match.group(2))
        else:
            fixed_point_repr = fixed_point_repr
            n_weights = group1


        # read the benchmark result file
        with open(f"{data_dir}/{filename}", "r") as f:
            lines = f.readlines()
            results =  [int(x) for x in lines]

        # go through results and write dicts
        for i, result in enumerate(results):
            d = {
                "repetition": i,
                "fixed_point_repr" : fixed_point_repr,
                "n_weights": n_weights,
                label: result
            }

            lst.append(d)

    return lst



def build_df_mbench_computation():

    df1 = _build_df_mbench_computation(data_dir="./data/microbenchmarks/large", cmachine="clientlarge", run_server=True)
    # df2 = _build_df_mbench_computation(data_dir="./data/microbenchmarks/small", cmachine="clientsmall", run_server=False)

    # df = df1.merge(df2)
    df = df1

    return df


def _build_df_mbench_computation(data_dir, cmachine, run_server):

    # 1st loop over all files in folder
    l1 = []
    l2 = []
    l3 = []
    l4 = []
    l5 = []
    l6 = []
    l7 = []
    l8 = []
    l9 = []
    l10 = []
    l11 = []
    l12 = []
    l13 = []
    l14 = []

    for filename in os.listdir(data_dir):

        # extract randproof
        pattern = "create-randproof-([0-9]+)-([0-9]+)-.*.bench"
        lst1 = extract(filename=filename, pattern=pattern, label="create_randproof_ms", data_dir=data_dir)
        l1 += lst1

        if run_server:
            pattern = "verify-randproof-([0-9]+)-([0-9]+)-.*.bench"
            lst2 = extract(filename=filename, pattern=pattern, label="verify_randproof_ms", data_dir=data_dir)
            l2 += lst2

        # extract rangeproof
        pattern = "create-rangeproof-([0-9]+)-[0-9]+-([0-9]+)-.*.bench"
        lst3 = extract(filename=filename, pattern=pattern, label="create_rangeproof_ms", data_dir=data_dir)
        l3 += lst3

        if run_server:
            pattern = "verify-rangeproof-part36-([0-9]+)-[0-9]+-([0-9]+)-.*.bench"
            lst4 = extract(filename=filename, pattern=pattern, label="verify_rangeproof_ms", data_dir=data_dir)
            l4 += lst4

        # extract rangeproof l2
        pattern = "create-rangeproof-l2-([0-9]+)-[0-9]+-([0-9]+)-.*.bench"
        lst5 = extract(filename=filename, pattern=pattern, label="create_rangeproofl2_ms", data_dir=data_dir)
        l5 += lst5

        if run_server:
            pattern = "verify-rangeproof-l2-([0-9]+)-[0-9]+-([0-9]+)-.*.bench"
            lst6 = extract(filename=filename, pattern=pattern, label="verify_rangeproofl2_ms", data_dir=data_dir)
            l6 += lst6

        # extract squarerandproof
        pattern = "create-squarerandproof-([0-9]+)-([0-9]+)-.*.bench"
        lst7 = extract(filename=filename, pattern=pattern, label="create_squarerandproof_ms", data_dir=data_dir)
        l7 += lst7

        if run_server:
            pattern = "verify-squarerandproof-([0-9]+)-([0-9]+)-.*.bench"
            lst8 = extract(filename=filename, pattern=pattern, label="verify_squarerandproof_ms", data_dir=data_dir)
            l8 += lst8

        if run_server:
            # extract discrete log
            pattern = "bench_paper_dlog2-([0-9]+)-65536-([0-9]+)-.*.bench"
            lst9 = extract(filename=filename, pattern=pattern, label="dlog2_ms", data_dir=data_dir)
            l9 += lst9

            # extract el gamal addition
            pattern = "bench_paper_addelgamal-([0-9]+)-.*.bench"
            lst10 = extract(filename=filename, pattern=pattern, label="elgamal_add_ms", data_dir=data_dir, fixed_point_repr=16)
            l10 += lst10

        pattern = "create-squareproof-([0-9]+)-([0-9]+)-.*.bench"
        lst11 = extract(filename=filename, pattern=pattern, label="create_squareproof_ms", data_dir=data_dir)
        l11 += lst11

        if run_server:
            pattern = "verify-squareproof-([0-9]+)-([0-9]+)-.*.bench"
            lst12 = extract(filename=filename, pattern=pattern, label="verify_squareproof_ms", data_dir=data_dir)
            l12 += lst12

        pattern = "create-compressedrandproof-([0-9]+)-([0-9]+)-.*.bench"
        lst13 = extract(filename=filename, pattern=pattern, label="create_compressedrandproof_ms", data_dir=data_dir)
        l13 += lst13

        if run_server:
            pattern = "verify-compressedrandproof-([0-9]+)-([0-9]+)-.*.bench"
            lst14 = extract(filename=filename, pattern=pattern, label="verify_compressedrandproof_ms", data_dir=data_dir)
            l14 += lst14



    # Combine Data from Different Experiments in separate Columns
    df = pd.DataFrame(l1)
    df = df.merge(pd.DataFrame(l3), how="outer")
    df = df.merge(pd.DataFrame(l5), how="outer")
    df = df.merge(pd.DataFrame(l7), how="outer")
    df = df.merge(pd.DataFrame(l11), how="outer")
    df = df.merge(pd.DataFrame(l13), how="outer")

    if run_server:
        df = df.merge(pd.DataFrame(l2), how="outer")
        df = df.merge(pd.DataFrame(l4), how="outer")
        df = df.merge(pd.DataFrame(l6), how="outer")
        df = df.merge(pd.DataFrame(l8), how="outer")
        df = df.merge(pd.DataFrame(l9), how="outer")
        df = df.merge(pd.DataFrame(l10), how="outer")
        df = df.merge(pd.DataFrame(l12), how="outer")
        df = df.merge(pd.DataFrame(l14), how="outer")


    df = df.sort_values(["fixed_point_repr", "n_weights", "repetition"])


    agg_client_d = {
        'create_randproof_ms' :['mean', 'var', 'count'],
        'create_rangeproof_ms' :['mean', 'var', 'count'],
        'create_rangeproofl2_ms' :['mean', 'var', 'count'],
        'create_squarerandproof_ms' :['mean', 'var', 'count'],
        'create_squareproof_ms' :['mean', 'var', 'count'],
        'create_compressedrandproof_ms' :['mean', 'var', 'count'],
    }

    agg_server_d = {
        'verify_randproof_ms' :['mean', 'var', 'count'],
        'verify_rangeproof_ms' :['mean', 'var', 'count'],
        'verify_rangeproofl2_ms' :['mean', 'var', 'count'],
        'verify_squarerandproof_ms' :['mean', 'var', 'count'],
        'verify_squareproof_ms' :['mean', 'var', 'count'],
        'verify_compressedrandproof_ms' :['mean', 'var', 'count'],
        'dlog2_ms': ['mean', 'var', 'count'],
        'elgamal_add_ms': ['mean', 'var', 'count']
    }

    if run_server:
        agg_d = {**agg_client_d, **agg_server_d}
    else:
        agg_d = agg_client_d


    # Aggregate Repetitions into Mean and variance
    df = df.groupby(["fixed_point_repr", "n_weights"], as_index=False).agg(agg_d)

    # convert to flat df
    df.columns = ['_'.join(tup).rstrip('_') for tup in df.columns.values]

    # Construct relevant metrics

    # client wellformedness
    df[f"l2_{cmachine}_wellformed_ms"] = df["create_squarerandproof_ms_mean"]
    df[f"l2opt_{cmachine}_wellformed_ms"] = df["create_squareproof_ms_mean"] + df["create_compressedrandproof_ms_mean"] # TODO: squareproof + compressedrandproof
    df[f"l8_{cmachine}_wellformed_ms"] = df["create_randproof_ms_mean"]
    df[f"l8p_{cmachine}_wellformed_ms"] = df["create_compressedrandproof_ms_mean"] # TODO: compressedrandproof

    df[f"l2_{cmachine}_wellformed_ms_std"] = np.sqrt(df["create_squarerandproof_ms_var"])
    df[f"l2opt_{cmachine}_wellformed_ms_std"] = np.sqrt(df["create_squareproof_ms_var"] + df["create_compressedrandproof_ms_var"]) # TODO: squareproof + compressedrandproof
    df[f"l8_{cmachine}_wellformed_ms_std"] = np.sqrt(df["create_randproof_ms_var"])
    df[f"l8p_{cmachine}_wellformed_ms_std"] = np.sqrt(df["create_compressedrandproof_ms_var"]) # TODO: compressedrandproof

    # server wellformedness
    if run_server:
        df["l2_server_wellformed_ms"] = df["verify_squarerandproof_ms_mean"]
        df["l2opt_server_wellformed_ms"] = df["verify_squareproof_ms_mean"] + df["verify_compressedrandproof_ms_mean"] # TODO: squareproof + compressedrandproof
        df["l8_server_wellformed_ms"] = df["verify_randproof_ms_mean"]
        df["l8p_server_wellformed_ms"] = df["verify_compressedrandproof_ms_mean"] # TODO: compressedrandproof

        df["l2_server_wellformed_ms_std"] = np.sqrt(df["verify_squarerandproof_ms_var"])
        df["l2opt_server_wellformed_ms_std"] = np.sqrt(df["verify_squareproof_ms_var"] + df["verify_compressedrandproof_ms_var"])
        df["l8_server_wellformed_ms_std"] = np.sqrt(df["verify_randproof_ms_var"])
        df["l8p_server_wellformed_ms_std"] = np.sqrt(df["verify_randproof_ms_var"])

    # client range
    df[f"l2_{cmachine}_range_ms"] = df["create_rangeproofl2_ms_mean"] + df["create_rangeproof_ms_mean"]
    df[f"l2opt_{cmachine}_range_ms"] = df["create_rangeproofl2_ms_mean"] + df["create_rangeproof_ms_mean"]
    df[f"l8_{cmachine}_range_ms"] = df["create_rangeproof_ms_mean"]
    df[f"l8p_{cmachine}_range_ms"] = df[df["n_weights"] == 8192]["create_rangeproof_ms_mean"].values[0]

    df[f"l2_{cmachine}_range_ms_std"] = np.sqrt(df["create_rangeproofl2_ms_var"] + df["create_rangeproof_ms_var"])
    df[f"l2opt_{cmachine}_range_ms_std"] = np.sqrt(df["create_rangeproofl2_ms_var"] + df["create_rangeproof_ms_var"])
    df[f"l8_{cmachine}_range_ms_std"] = np.sqrt(df["create_rangeproof_ms_var"])
    df[f"l8p_{cmachine}_range_ms_std"] = np.sqrt(df[df["n_weights"] == 8192]["create_rangeproof_ms_var"].values[0])

    # server range
    if run_server:
        df["l2_server_range_ms"] = df["verify_rangeproofl2_ms_mean"] + df["verify_rangeproof_ms_mean"]
        df["l2opt_server_range_ms"] = df["verify_rangeproofl2_ms_mean"] + df["verify_rangeproof_ms_mean"]
        df["l8_server_range_ms"] = df["verify_rangeproof_ms_mean"]
        df["l8p_server_range_ms"] = df[df["n_weights"] == 8192]["verify_rangeproof_ms_mean"].values[0]

        df["l2_server_range_ms_std"] = np.sqrt(df["verify_rangeproofl2_ms_var"] + df["verify_rangeproof_ms_var"])
        df["l2opt_server_range_ms_std"] = np.sqrt(df["verify_rangeproofl2_ms_var"] + df["verify_rangeproof_ms_var"])
        df["l8_server_range_ms_std"] = np.sqrt(df["verify_rangeproof_ms_var"])
        df["l8p_server_range_ms_std"] = np.sqrt(df[df["n_weights"] == 8192]["verify_rangeproof_ms_var"].values[0])

    # el gamal aggregation
    if run_server:
        df["l2_server_caggregation_ms"] = df["elgamal_add_ms_mean"]
        df["l2opt_server_caggregation_ms"] = df["elgamal_add_ms_mean"]
        df["l8_server_caggregation_ms"] = df["elgamal_add_ms_mean"]
        df["l8p_server_caggregation_ms"] = df["elgamal_add_ms_mean"]

        df["l2_server_caggregation_ms_std"] = np.sqrt(df["elgamal_add_ms_var"])
        df["l2opt_server_caggregation_ms_std"] = np.sqrt(df["elgamal_add_ms_var"])
        df["l8_server_caggregation_ms_std"] = np.sqrt(df["elgamal_add_ms_var"])
        df["l8p_server_caggregation_ms_std"] = np.sqrt(df["elgamal_add_ms_var"])

    # log2 reconstruction
    if run_server:
        df["server_log2reconstruct_ms"] = df["dlog2_ms_mean"]
        df["server_log2reconstruct_ms_std"] = np.sqrt(df["dlog2_ms_var"])


    # filter out the probabilistic checking
    df = df[df["n_weights" ] != 8192]

    # project to new columns
    cols = ["n_weights", "fixed_point_repr"]
    new_cols = [col for col in df.columns.values if col.startswith("l2_") or col.startswith("l2opt_") or col.startswith("l8_") or col.startswith("l8p_")]
    cols += new_cols
    if run_server:
        cols += ["server_log2reconstruct_ms", "server_log2reconstruct_ms_std"]
    df = df[cols]


    return df


def format_plot_computation(ax, ind, width, group_labels, ytick_step=50):
    ##########################
    # General Format
    ##########################
    # ax.set_title("Hello World")

    def legend_without_duplicate_labels(ax):
        handles, labels = ax.get_legend_handles_labels()
        unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
        ax.legend(*zip(*unique), loc="best") # 'best', 'upper right', 'upper left', 'lower left',
        # 'lower right', 'right', 'center left',  'center right',
        # 'lower center', 'upper center', 'center'
    legend_without_duplicate_labels(ax)

    ax.grid(True, axis="y", linestyle=':', color='0.6', zorder=0, linewidth=1.2)

    ##########################
    # Y - Axis Format
    ##########################
    ax.set_ylim(ymin=0, ymax=None)
    ax.set_ylabel("Time [s]")

    ymax = ax.get_ylim()[1]
    ax.set_yticks(np.arange(0, ymax, ytick_step))

    ##########################
    # X - Axis Format
    ##########################
    offsets = np.arange(0, width * 4, width)
    offsets = offsets - (width * 4 / 2)
    xticks = []
    for i in range(offsets.shape[0]):
        xticks = np.append(xticks, ind + offsets[i])

    # # add 1st axis for norm (l2, l8, l8p)
    # xticks = np.append(ind, ind-width, axis=0)
    # xticks = np.append(xticks, ind+width, axis=0)
    xticks = np.sort(xticks)
    labels = 4 * ["$L_2^{}$", "$L_2^{(o)}$", "$L^{}_{\infty}$", "$L_{\infty}^{(o)}$"]
    ax.set_xticks(xticks)
    ax.set_xticklabels(labels, fontsize=14, rotation=345)


    # add 2nd axis for number of parameters
    ax2 = ax.twiny()
    ax2.set_xticks(ind)
    # $(2^{{{int(math.log(n_weights,2))}}})$
    group_labels = [f"{round(n_weights /1000)}k" for n_weights in group_labels]
    ax2.set_xticklabels(group_labels, rotation=345)
    ax2.xaxis.set_ticks_position('bottom')
    ax2.xaxis.set_label_position('bottom')
    ax2.spines['bottom'].set_position(('outward', 36))

    ax2.set_xlabel('Number of Parameters')
    ax2.set_xlim(ax.get_xlim())
    ax2.tick_params(axis='both', which='both', length=0)
    ax2.spines['bottom'].set_visible(False)


def build_fig_mbench_computation_server_perclient_zkp(df, name="mbench_computation_server_perclient_zkp"):
    setup_plt()

    with PdfPages(f"{output_dir}/{name}.pdf") as pdf:

        fig, ax = plt.subplots()

        ##########################
        # Draw all the lines
        ##########################

        # build ind, width
        ind = np.arange(0, len(df.index)) * 1.35
        width = 0.3  # the width of the bars
        offsets = np.arange(0, width * 4, width)
        offsets = offsets - (width * 4 / 2)


        # build the barcharts

        for offset, norm in zip(offsets, ["l2", "l2opt", "l8", "l8p"]):

            ax.bar(ind +offset, df[f"{norm}_server_range_ms"] / 1000, width,
                   bottom=(df[f"{norm}_server_wellformed_ms"] + df[f"{norm}_server_caggregation_ms"]) / 1000, label=range_config["server_label"],
                   color=range_config["color"], edgecolor="black", zorder=2)

            ax.bar(ind +offset, df[f"{norm}_server_wellformed_ms"] / 1000, width,
                   bottom=df[f"{norm}_server_caggregation_ms"] / 1000, label=wellformed_config["server_label"],
                   color=wellformed_config["color"], edgecolor="black", zorder=2)

            ax.bar(ind +offset, df[f"{norm}_server_caggregation_ms"] / 1000, width,
                   bottom=None, label=cagg_config["server_label"],
                   color=cagg_config["color"], edgecolor="black", zorder=2)



        # format the computation barchart
        format_plot_computation(ax, ind, width, df["n_weights"], 10)


        pdf.savefig(bbox_inches='tight', pad_inches=0)
        plt.close()
    return fig


def build_fig_mbench_computation_client_zkp(df, clientsuffix="small"):

    machine = f"client{clientsuffix}"
    name = f"mbench_computation_{machine}_zkp"

    setup_plt()

    with PdfPages(f"{output_dir}/{name}.pdf") as pdf:

        fig, ax = plt.subplots()

        ##########################
        # Draw all the lines
        ##########################

        # build ind, width
        ind = np.arange(0, len(df.index)) * 1.35
        width = 0.3  # the width of the bars
        offsets = np.arange(0, width * 4, width)
        offsets = offsets - (width * 4 / 2)


        # build the barcharts

        for offset, norm in zip(offsets, ["l2", "l2opt", "l8", "l8p"]):

            ax.bar(ind +offset, df[f"{norm}_{machine}_range_ms"] / 1000, width,
                   bottom=df[f"{norm}_{machine}_wellformed_ms"] / 1000, label=range_config["client_label"],
                   color=range_config["color"], edgecolor="black", zorder=2)

            ax.bar(ind +offset, df[f"{norm}_{machine}_wellformed_ms"] / 1000, width,
                   bottom=None, label=wellformed_config["client_label"],
                   color=wellformed_config["color"], edgecolor="black", zorder=2)


        if clientsuffix == "small":
            ytick_step =200
        else:
            ytick_step =50

        # format the computation barchart
        format_plot_computation(ax, ind, width, df["n_weights"], ytick_step=ytick_step)

        pdf.savefig(bbox_inches='tight', pad_inches=0)
        plt.close()
    return fig

def build_fig_mbench_computation_server_dlog(df, name="mbench_computation_server_dlog"):

    label = "discrete log reconstruction"
    color = "0.1"
    setup_plt()

    with PdfPages(f"{output_dir}/{name}.pdf") as pdf:

        fig, ax = plt.subplots()

        ##########################
        # Draw all the lines
        ##########################

        # build ind, width
        ind = np.arange(0, len(df.index)) * 1.2
        width = 0.3  # the width of the bars

        ax.bar(ind, df["server_log2reconstruct_ms"] / 1000, width,
               label=label, color=color, edgecolor="black", zorder=2)


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
        ax.set_ylim(ymin=0, ymax=0.8)
        ax.set_ylabel("Time [s]")
        # ax.set_yticks(yticks)
        # ax.set_yticklabels(labels, fontsize=16, rotation=345)


        ##########################
        # X - Axis Format
        ##########################
        # ax.set_xlim(xmin=0, xmax=None)
        ax.set_xlabel("Number of Parameters")
        ax.set_xticks(ind)
        labels = [f"{round( x /1000)}k" for x in df["n_weights"]]
        ax.set_xticklabels(labels)

        # labels = 4 * ["$L_2$", "$L_{\infty}$", "$L_{\infty}^{(p)}$"]
        # ax.set_xticks(xticks)
        # ax.set_xticklabels(labels, fontsize=16, rotation=345)

        pdf.savefig(bbox_inches='tight', pad_inches=0)
        plt.close()
    return fig
