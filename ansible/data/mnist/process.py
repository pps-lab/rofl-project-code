#%%

import numpy as np
import argparse
import tensorflow as tf

parser = argparse.ArgumentParser(description='Process dataset.')
parser.add_argument('--num_clients', type=int, default=48,
                    help='number of clients')

args = parser.parse_args()

#%% md

# Step 1: Configure parameters here

num_clients = args.num_clients
dataset = "mnist"

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

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train, x_test = x_train.astype(np.float32), x_test.astype(np.float32)
x_train, x_test = x_train[..., np.newaxis], x_test[..., np.newaxis]

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