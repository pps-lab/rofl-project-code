from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss
import pandas as pd
from tqdm.notebook import tqdm
import os, re, math
from plots.e2e_time import setup_plt

from common import get_colorful_styles, output_dir

def detection_prop_hypergeometric(n_weights, n_weights_violating_bound, n_bounds_check):
    # probability that by sampling {n_bounds_check} weights without replacement from all the {n_weights} weights
    # we get k=0 of the {n_weights_violating_bound} weights that violate the bound
    hpd = ss.hypergeom(n_weights, n_weights_violating_bound, n_bounds_check)
    k = 0
    p = hpd.pmf(k)

    # probability that we get at least one of the weights violating the bound
    return 1 - p


def binary_search_nchecks(success_prob_bound, n_weights, n_weights_violating_bound):
    mid = 0
    start = 0
    end = n_weights # 10000 # upper bound
    step = 0
    success = None
    while (start <= end):
        mid = (start + end) // 2

        prob = detection_prop_hypergeometric(n_weights=n_weights, n_weights_violating_bound=n_weights_violating_bound, n_bounds_check=mid)
        #print(f"Check mid={mid}  prob={prob}")

        if prob < success_prob_bound:
            # checking {mid} parameters not sufficient => increase the number of checks
            start = mid + 1
        else:
            # checking {mid} parameters, results in success probability above the bound => check if reducing number of checks still works
            success = (mid, prob)
            end = mid - 1

    if success is None:
        raise ValueError("did not find satisfiable number of checks")
    return success

def build_prob_checking_data():
    lines = []

    x_min = 1000
    x_max = 530000

    n_steps = 100
    step_size = int((x_max-x_min)/n_steps)

    colors, _ = get_colorful_styles()

    for fail_prob_bound, linestyle in tqdm(zip([1e-8], ['-'])): #, 1e-9

        success_prob_bound = 1 - fail_prob_bound
        for prob_weights_violating_bound, color in tqdm(zip([0.1, 0.01, 0.001], [colors[1], colors[2], colors[0]])):

            labels = range(x_min, x_max, step_size)
            values = []
            for n_weights in tqdm(labels, leave=False):
                n_weights_violating_bound = int(prob_weights_violating_bound * n_weights)
                s = binary_search_nchecks(success_prob_bound=success_prob_bound, n_weights=n_weights, n_weights_violating_bound=n_weights_violating_bound)
                n_checks = s[0]
                values.append(n_checks)
            d = {
                "fail_prob_bound": fail_prob_bound,
                "success_prob_bound": success_prob_bound,
                "prob_weights_violating_bound": prob_weights_violating_bound,
                #"n_weights_violating_bound": n_weights_violating_bound,
                "labels (n_weights)": labels,
                "values (n_checks)": values,
                "color": color,
                "linestyle": linestyle
            }
            lines.append(d)

    return lines

def build_fig_pcheck_num_required_checks(lines, name="pcheck_num_required_checks"):

    setup_plt()

    with PdfPages(f"{output_dir}/{name}.pdf") as pdf:

        fig, ax = plt.subplots()

        ##########################
        # Draw all the lines
        ##########################

        # lineplot
        for line in lines:

            fail_prob_bound = line["fail_prob_bound"]
            prob_weights_violating_bound = line["prob_weights_violating_bound"]
            label = f"$\delta = {fail_prob_bound}$    $p_v = {prob_weights_violating_bound}$"
            plt.plot(line["labels (n_weights)"], line["values (n_checks)"], label=label, color=line["color"], linestyle=line["linestyle"], linewidth=2)

        ##########################
        # General Format
        ##########################
        #ax.set_title("Hello World")
        ax.legend(loc="best")   # 'best', 'upper right', 'upper left', 'lower left',
        # 'lower right', 'right', 'center left',  'center right',
        # 'lower center', 'upper center', 'center'
        ax.grid(True, axis="y", linestyle=':', color='0.6', zorder=0, linewidth=1.2)

        ##########################
        # Y - Axis Format
        ##########################
        ax.set_ylim(ymin=0, ymax=None)
        ax.set_ylabel("Required Checks")
        #ax.set_yticks(yticks)
        ylabels = [f"{round(y)}k" for y in ax.get_yticks()/1000]
        ax.set_yticklabels(ylabels)

        ##########################
        # X - Axis Format
        ##########################
        ax.set_xlim(xmin=0, xmax=None)
        ax.set_xlabel("Number of Parameters")
        #ax.set_xticks(xticks)
        xlabels = [f"{round(x)}k" for x in ax.get_xticks()/1000]
        ax.set_xticklabels(xlabels)

        pdf.savefig(bbox_inches='tight', pad_inches=0)
        plt.close()
    return fig


# def build_prob_check_attack_success():
#
#     mal_update_size = 44426
#     mal_update_outside_bound = 5004
#
#     mal_update = np.zeros([mal_update_size], dtype=np.bool)
#     mal_update[:mal_update_outside_bound] = True
#     mal_update = np.random.permutation(mal_update)
#     print(mal_update)
#
#     p_vs = np.arange(0.01, 1.0, 10)
#
#     for p_v in p_vs:
#         # we select p_v at random
#
#
#     lines []
#     x_min = 1000
#     x_max = 530000
#
#     n_steps = 100
#     step_size = int((x_max-x_min)/n_steps)
#
#     colors, _ = get_colorful_styles()
#
#     for fail_prob_bound, linestyle in tqdm(zip([1e-8], ['-'])): #, 1e-9
#
#         success_prob_bound = 1 - fail_prob_bound
#         for prob_weights_violating_bound, color in tqdm(zip([0.1, 0.01, 0.001], [colors[1], colors[2], colors[3]])):
#
#             labels = range(x_min, x_max, step_size)
#             values = []
#             for n_weights in tqdm(labels, leave=False):
#                 n_weights_violating_bound = int(prob_weights_violating_bound * n_weights)
#                 s = binary_search_nchecks(success_prob_bound=success_prob_bound, n_weights=n_weights, n_weights_violating_bound=n_weights_violating_bound)
#                 n_checks = s[0]
#                 values.append(n_checks)
#             d = {
#                 "fail_prob_bound": fail_prob_bound,
#                 "success_prob_bound": success_prob_bound,
#                 "prob_weights_violating_bound": prob_weights_violating_bound,
#                 #"n_weights_violating_bound": n_weights_violating_bound,
#                 "labels (n_weights)": labels,
#                 "values (n_checks)": values,
#                 "color": color,
#                 "linestyle": linestyle
#             }
#             lines.append(d)
#
#     return lines