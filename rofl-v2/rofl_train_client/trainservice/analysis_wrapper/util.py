
import numpy as np

def unflatten(w, weights):
    sizes = [x.size for x in weights]
    split_idx = np.cumsum(sizes)
    update_ravelled = np.split(w, split_idx)[:-1]
    shapes = [x.shape for x in weights]
    update_list = [np.reshape(u, s) for s, u in zip(shapes, update_ravelled)]
    return update_list

def flatten_update(update):
    return np.concatenate([x.ravel() for x in update])