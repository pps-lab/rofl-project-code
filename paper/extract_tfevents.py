"""

Helper file to extract df directly from tfevents files

This removes the requirement to first extract using command line

"""

import os
from os.path import isfile, isdir

import pandas as pd
import numpy as np
import re

from aggregator import aggregate

EXPERIMENTS_PATH = "data/raw_experiments"

def create_df_scaling_factor_cifar(prefix, directory):

    # prefix = "e1_cifar_resnet18_scaling_factor_"
    task_translation = {
        "WALL": "a2-wall",
        "GREEN": "a3-green",
        "STRIPES": "a4-stripes"
    }
    SCALING_FACTORS = {
        40: [1, 10, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 90, 100]
    }

    df = pd.DataFrame(SCALING_FACTORS[40], columns=["scaling_factor"])
    df["n_clients"] = 40

    for attack_task in task_translation.keys():
        PATH = os.path.join(EXPERIMENTS_PATH, directory, f"{prefix}{attack_task}")
        df1 = aggregate(PATH, 'df', ['events'], ["run-*"],
                    ['evaluation/test_accuracy', 'evaluation/adv_success', 'l2_total/mal'])

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
        df = pd.concat([df, df_sorted], axis=1)
        # print(df)
    #     if n_clients == 10:
    #         df_10 = pd.concat([df_10, df_sorted], axis=1)
    #     elif n_clients == 20:
    #         df_20 = pd.concat([df_20, df_sorted], axis=1)
    #     elif n_clients == 40:
    #         df_40  = pd.concat([df_40, df_sorted], axis=1)
    #     else:
    #         print(f"Ignore file: {filename} with n_clients={n_clients}")
    #
    #
    # df = pd.concat([df_10, df_20, df_40])

    df["alpha_fracadv"] = 1 / df["n_clients"]
    return df


def create_df_benign_norms_statistics(directory):
    PATH = os.path.join(EXPERIMENTS_PATH, directory)
    files = [f for f in os.listdir(PATH) if isfile(os.path.join(PATH, f))]
    df = pd.DataFrame([], columns=[])

    for file in files:
        pattern = f"round_([0-9]+).npy"
        match = re.search(pattern, file, re.IGNORECASE)
        if match:
            round_number = match.group(1)
            round_file = np.load(os.path.join(PATH, file), allow_pickle=True)
            benign_norms_l2, benign_norms_l1, mal_norms_l2, mal_norms_l1 =\
                np.array(round_file[0]), np.array(round_file[1]), np.array(round_file[2]), np.array(round_file[3])

            # In this round, the attacker does not attack so behaves benign
            benign_norms_l2 = np.append(benign_norms_l2, mal_norms_l2[0])

            min = benign_norms_l2.min()
            max = benign_norms_l2.max()
            median = np.median(benign_norms_l2)
            mean = np.mean(benign_norms_l2)

            df_2 = pd.DataFrame([[round_number, min, max, median, mean]], columns=['Round', 'min', 'max', 'median', 'mean'], index=[round_number])
            df = df.append(df_2)

    return df


def create_df_directory_test_adv(directory):
    PATH = os.path.join(EXPERIMENTS_PATH, directory)

    df = pd.DataFrame(list(range(1000)), columns=["Round"])

    df1 = aggregate(PATH, 'df', ['events'], ["e63_*"],
                    ['evaluation/test_accuracy', 'evaluation/adv_success'])

    df = pd.concat([df, df1], axis=1)

    return df


create_df_directory_test_adv("edge_cases")

# create_df_benign_norms_statistics("scaling_factor/benign_norms")

# create_df_scaling_factor_resnet18()