# This file contains common setup code for plot formatting, and data loading from the data source.

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import pymongo
import matplotlib

output_dir = "plots_output"
COLOR_GRAY = "#999999"

client = None
data_source = "mongodb"  # allowed values: mongodb, json


def _query_mongo(query):
    global client
    if client is None:
        client = pymongo.MongoClient("mongodb://localhost:27017/")

    db = client["analysis"]
    docs = db['experiments'].find(query)
    print(docs)
    results = list(docs)
    print([r["_id"] for r in results])
    return results


def query_data(query):
    if data_source == "mongodb":
        return _query_mongo(query)
    elif data_source == "json":
        raise NotImplementedError("JSON source not yet supported")
    else:
        raise ValueError(f"Data source {data_source} not supported.")


def _preprocess(df, suffix):
    # set start timestamp to 0
    # df["time"].iloc[0] = 1
    # df["cumtime"] = df["time"].cumsum()
    df = df.rename(columns={"adv_success": f"adv_success_{suffix}", "accuracy": f"accuracy_{suffix}"})
                            # "time": f"time_{suffix}", "cumtime": f"cumtime_{suffix}" }
                   # )
    return df

def get_colorful_styles():
    cmap_1 = matplotlib.cm.get_cmap('Set1')
    cmap_2 = matplotlib.cm.get_cmap('Set2')
    # colors = [cmap_1(i) for i in range(8)]
    colors = []
    colors.extend([cmap_1(i) for i in range(30)])
    # colors = ['#CD4631', '#8B1E3F', '#3C153B', '#89BD9E', '#F9C80E', '#F9C80E', '#F9C80E', '#F9C80E']
    linestyles = ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-',
                  '-', '-', '-', '-', '-', '-', '-', '-', '-', '-']

    colors = [colors[3], colors[1], colors[2], colors[0]] + colors[4:]

    return colors, linestyles


def get_markers():
    return ['*', 'o', 'D', 's', 'p', 'h', '']


def setup_plt(square=False):

    fig_width_pt = 240.0  # Get this from LaTeX using \showthe
    inches_per_pt = 1.0 / 72.27 * 2  # Convert pt to inches
    golden_mean = ((np.math.sqrt(5) - 1.0) / 2.0) * .8  # Aesthetic ratio
    fig_width = fig_width_pt * inches_per_pt  # width in inches
    fig_height = (fig_width * golden_mean)  # height in inches
    fig_size =  [fig_width, fig_height]

    if square:
        fig_size = [fig_height * 1.58, fig_height]
        plt_params = {
            'backend': 'ps',
            'axes.labelsize': 18,
            'legend.fontsize': 14,
            'xtick.labelsize': 16,
            'ytick.labelsize': 16,
            'font.size': 16,
            'figure.figsize': fig_size,
            'font.family': 'Times New Roman'
        }
    else:
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


def yaxis_formatter(x, p):
    return int(round(x * 100))