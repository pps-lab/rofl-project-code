
import pandas as pd
import os
import re
import time
import datetime


def load_benchlog(run_dir):

    benchlog_dir = os.path.join(run_dir, "benchlog")

    data_clients = pd.DataFrame()

    # load_clients_benchlog(benchlog_dir, data_clients)

    ## Load server
    header_list_server = ["round", "start", "agg_done",
                          "param_extraction_done", "verification_complete"]

    bench_dir = os.path.join(benchlog_dir, "server/benchlog")
    all_files = os.listdir(bench_dir)
    sort = sorted(all_files)
    # print(sort)
    first_file = sort[-1]

    data_server = pd.read_csv(os.path.join(bench_dir, first_file), delimiter=",", names=header_list_server)

    observer_data = load_observer_output(os.path.join(run_dir, "logs", "server", "observer_stdout.log"))

    return data_server, observer_data


def load_clients_benchlog(benchlog_dir, data_clients):
    ## Load clients
    pattern = "^client_(\\d*)$"
    header_list_clients = ["round", "model_meta_received", "model_completely_received",
                           "local_model_training_done", "model_enc_proofs_dome", "model_sent"]
    for filename in os.listdir(benchlog_dir):

        match = re.search(pattern, filename)
        if not match:
            continue

        client_id = int(match.group(1))

        content = pd.read_csv(os.path.join(benchlog_dir, filename), delimiter=",", header=header_list_clients)
        content["client"] = client_id

        data_clients.append(content)


def load_observer_output(observer_file):
    with open(observer_file, 'r') as f:
        # Read the file contents and generate a list with each line
        lines = f.readlines()

    data = pd.DataFrame(columns=['time', 'round', 'loss', 'acc'])

    # Iterate each line
    for line in lines:

        # Regex applied to each line
        match = re.search("(.*) - root - INFO - \[EVAL\] Test \(round,loss,accuracy\): \((\d*), (.*), ([\d.]*)\)", line)
        if match:
            # Make sure to add \n to display correctly when we write it back
            time_str = match.group(1)
            round = int(match.group(2))
            loss = float(match.group(3))
            accuracy = float(match.group(4))
            dt = datetime.datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S,%f")
            timestamp = time.mktime(datetime.datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S,%f").timetuple())

            data = data.append({'time': dt, 'round': round, 'loss': loss, 'acc': accuracy}, ignore_index=True)

    return data

