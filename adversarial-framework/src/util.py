import collections
from copy import deepcopy

import numpy as np
import pandas as pd

from os.path import join
from tensorflow.python.keras.layers.convolutional import Conv2D
from tensorflow.python.keras.layers.core import Dense


def log_data(experiment_dir, rounds, accuracy, adv_success):
    """Logs data."""
    df = pd.DataFrame()
    df['round'] = rounds
    df['accuracy'] = accuracy
    df['adv_success'] = adv_success
    df.to_csv(join(experiment_dir, 'log.csv'), index=False)


def power_iteration(A):
    """Computes principle eigenvalue and eigenvector.

    Args:
        A (np.ndarray): Square matrix.

    Returns:
        tuple: Tuple of eigenvalue and eigenvector of np.ndarray type.
    """

    def eigenvalue(A, v):
        Av = A.dot(v)
        return v.dot(Av)

    n, d = A.shape

    v = np.ones(d) / np.sqrt(d)
    ev = eigenvalue(A, v)

    while True:
        Av = A.dot(v)
        v_new = Av / np.linalg.norm(Av)

        ev_new = eigenvalue(A, v_new)
        if np.abs(ev - ev_new) < 0.01:
            break

        v = v_new
        ev = ev_new

    return ev_new, v_new


def create_dropout_mask(model, federated_dropout_rate, federated_dropout_all_parameters, n_clients=1):
    """Applies dropout on model parameters as described in:
    Caldas, S., Konečny, J., McMahan, H.B. and Talwalkar, A., 2018. Expanding the Reach of Federated Learning by
    Reducing Client Resource Requirements. arXiv preprint arXiv:1812.07210

    The fixed number of neurons for Dense layers (and filters for Conv2D layer) are zeroed out expect for
    very first and last layers (unless federated_dropout_all_parameters is True). Biases are intact.

    Args:
        model (tf.model): Keras model.
        federated_dropout_rate (float): Federated dropout rate in (0, 1) range.
        federated_dropout_all_parameters (bool): Program parameter.
        n_clients (int): How many non-overlapping dropout masks to create.

    Returns:
        For each client a list of np.ndarray that represent dropout mask.
    """
    assert 0 < federated_dropout_rate < 1., 'Federated dropout rate must be in (0, 1) range.'
    assert type(model.layers[0]) in [Conv2D, Dense], \
        "The implementation assumes that the first layer is Dense or Conv2D"

    layer_range = 1, len(model.layers) - 1
    if federated_dropout_all_parameters:
        layer_range = 0, len(model.layers)

    dropout_mask = [[np.ones_like(l, dtype=bool) for l in model.get_weights()] for _ in range(n_clients)]

    # elems_to_drop = 1.0 - federated_dropout_rate

    layer_ind = layer_range[0] * 2  # since we skip the first layer
    for ind in range(layer_range[0], layer_range[1]):
        if type(model.layers[ind]) in [Conv2D, Dense]:
            param_shape = model.layers[ind].weights[0].shape
            if federated_dropout_all_parameters:  # partially zeroed out filters
                assert n_clients * federated_dropout_rate < 1
                # param_shape = (kernel w, kernel h, prev layer filters, current layer filters)
                total_params = np.prod(param_shape)
                n_select = int(federated_dropout_rate * total_params) * n_clients
                keep_inds = np.random.choice(total_params, n_select, replace=False)
                keep_inds = keep_inds.reshape((n_clients, -1))

                for client in range(n_clients):
                    layer_mask = np.zeros(np.prod(param_shape), dtype=bool)
                    layer_mask[keep_inds[client]] = True
                    dropout_mask[client][layer_ind] = layer_mask.reshape(param_shape)

            else:
                n_select = int(federated_dropout_rate * param_shape[-1]) * n_clients
                keep_inds = np.random.choice(param_shape[-1], n_select, replace=True)
                keep_inds = keep_inds.reshape((n_clients, -1))

                for client in range(n_clients):
                    layer_mask = np.zeros_like(dropout_mask[client][layer_ind], dtype=bool)
                    layer_mask[..., keep_inds[client]] = True
                    dropout_mask[client][layer_ind] = layer_mask

            layer_ind += 2  # ind*2 because we zero out only weights (not biases)

    return dropout_mask

def aggregate_weights_masked(current_weights, global_learning_rate, num_clients, dropout_rate, client_dropout_mask, client_weight_list):
    """Procedure for merging client weights together with `global_learning_rate`."""

    assert len(current_weights) == len(client_weight_list[0])
    assert len(client_dropout_mask) == len(client_weight_list)
    assert len(client_dropout_mask[0]) == len(client_weight_list[0])

    new_weights = deepcopy(current_weights)
    number_of_clients_participating_this_round = len(client_dropout_mask)
    # Estimate impact of this update
    update_coefficient = global_learning_rate / num_clients
    client_weight_list_masked = []
    for mask, w in zip(client_dropout_mask, client_weight_list):
        client = []
        for mask_l, w_l, old_w_l in zip(mask, w, current_weights):
            update = w_l - old_w_l
            update[mask_l == False] = float('nan')
            client.append(update)
        client_weight_list_masked.append(client)

    client_weight_list_t = [list(i) for i in zip(*client_weight_list_masked)]
    update_weight_list = [np.nan_to_num(np.nansum(w, axis=0)) for w in client_weight_list_t]
    counts = [np.sum(np.array(list(i), dtype=np.int), axis=0) for i in zip(*client_dropout_mask)]
    update_weight_list = [update_coefficient * w for w, c in zip(update_weight_list, counts)]
    for layer in range(len(current_weights)):
        new_weights[layer] = new_weights[layer] + \
                             update_weight_list[layer]
    return new_weights


def flatten(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
