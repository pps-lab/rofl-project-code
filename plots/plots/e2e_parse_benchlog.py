
import pandas as pd
import os
import re
import time
import datetime


def load_benchlog(run_dir):

    benchlog_dir = os.path.join(run_dir, "benchlog")

    data_clients = load_clients_benchlog(benchlog_dir)

    data_server = load_server_benchlog(benchlog_dir)

    observer_data = load_observer_output(os.path.join(run_dir, "logs", "server", "observer_stdout.log"))

    return data_server, observer_data, data_clients


def load_server_benchlog(benchlog_dir):
    ## Load server
    header_list_server = ["round", "start", "agg_done",
                          "param_extraction_done", "verification_complete"]
    bench_dir = os.path.join(benchlog_dir, "server/benchlog")
    all_files = os.listdir(bench_dir)
    sort = sorted(all_files)
    # print(sort)
    first_file = sort[-1]
    data_server = pd.read_csv(os.path.join(bench_dir, first_file), delimiter=",", names=header_list_server)
    return data_server


def load_clients_benchlog(benchlog_dir):
    ## Load clients
    pattern = "^client_(\\d*)$"
    header_list_clients = ["round", "model_download", "model_training",
                           "model_crypto", "model_upload", "total_duration",
                           "bytes_received", "bytes_sent"]

    data_clients = pd.DataFrame()
    for filename in os.listdir(benchlog_dir):

        match = re.search(pattern, filename)
        if not match:
            continue

        machine_id = int(match.group(1))

        client_dir = os.path.join(benchlog_dir, f"client_{machine_id}")
        all_files = os.listdir(client_dir)
        sort = sorted(all_files)
        first_file = sort[-1]

        content = pd.read_csv(os.path.join(client_dir, first_file), delimiter=",", names=header_list_clients)
        content["machine_id"] = machine_id

        data_clients = data_clients.append(content)

    return data_clients


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

