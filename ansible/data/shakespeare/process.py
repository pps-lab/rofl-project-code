#%%

import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Process dataset.')
parser.add_argument('--num_clients', type=int, default=48,
                    help='number of clients')

args = parser.parse_args()

#%% md

# Step 1: Configure parameters here

num_clients = args.num_clients
dataset = "shakespeare"

import glob, os
from pathlib import Path

dtype_file = Path(f"../dataset_type_{dataset}.txt")
if dtype_file.exists():
    print("Dataset already loaded and ready!")
    exit(0)
else:
    files = glob.glob('../dataset_type_*.txt')
    # Iterate over the list of files and remove individually
    for file in files:
        print(f"Removing {file}")
        os.remove(file)

    dtype_file.touch()


# Download shakespeare dataset

import subprocess
list_files = subprocess.run(["ls", "-l", "leaf/shakespeare/data"])
subprocess.run(["rm", "-r", "leaf/shakespeare/data/"])
subprocess.run(["./preprocess.sh", "-s", "niid", "--sf", "0.2", "-k", "0", "-t", "sample", "-tf", "0.8"], cwd="leaf/shakespeare")

# !rm -r leaf/shakespeare/data/*
# !cd leaf/shakespeare && ./preprocess.sh -s niid --sf 0.2 -k 0 -t sample -tf 0.8

#%% sh
subprocess.run(["./stats.sh"], cwd="leaf/shakespeare")

#%% md
# Step 2:

#%%
from leaf_loader import load_leaf_dataset, process_text_input_indices, process_char_output_indices

print("Loading files into python")

users, train_data, test_data = load_leaf_dataset("shakespeare")

x_train = [process_text_input_indices(train_data[user]['x']) for user in users]
y_train = [process_char_output_indices(train_data[user]['y']) for user in users]

x_train = np.concatenate(x_train)
y_train = np.concatenate(y_train)

x_test = np.concatenate([process_text_input_indices(test_data[user]['x']) for user in users])
y_test = np.concatenate([process_char_output_indices(test_data[user]['y']) for user in users])

# Step 3: Save the results.

#%%

# Sort IID
perms = np.random.permutation(x_train.shape[0])
x_train, y_train = x_train[perms, :], y_train[perms]

for client_id in range(num_clients):
    data_samples = int(x_train.shape[0] / num_clients)
    inds = (client_id * data_samples, (client_id + 1) * data_samples)
    x, y = x_train[inds[0]:inds[1]], y_train[inds[0]:inds[1]]
    np.save(f"../dataset_{client_id}", (x, y, x_test, y_test))

print("Saved files successfully!")