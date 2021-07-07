#!/usr/bin/env python3


# Get results from analysis run
import pandas as pd
import os
import sys
from aggregator import aggregate

def parse(run_dir):

    # try to find a log.csv
    path = os.path.join(run_dir, "log.csv")
    if os.path.isfile(path):
        csv_pd = pd.read_csv(os.path.join(run_dir, "log.csv"))
        return csv_pd.to_json()

    # try extracting the tfevents file
    # print(run_dir)
    path_events = os.path.join(run_dir, 'events')
    if not os.path.exists(path_events):
        return '{"result": null}'
    df1 = aggregate(run_dir, 'df', ['events'], [],
                    ['evaluation/test_accuracy', 'evaluation/adv_success'])
    df1.rename(columns={'0_evaluation/test_accuracy': 'accuracy',
                        '0_evaluation/adv_success': 'adv_success'}, inplace=True)
    df1['round'] = range(1, len(df1) + 1)

    # print(df1.to_json())

    return df1.to_json()


print(parse(sys.argv[1]))