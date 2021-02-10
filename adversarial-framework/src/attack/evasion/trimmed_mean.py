
from .evasion_method import EvasionMethod

import numpy as np
import tensorflow as tf

class TrimmedMeanEvasion(EvasionMethod):

    def __init__(self, benign_updates_this_round, alpha, n_remove_malicious):
        """

        :type benign_updates_this_round: [[np.ndarray]] list of client updates,
        :type alpha: float alpha of regularization
        :type n_remove_malicious int number of updates to trim on each side
        one update contains a list of per-layer weights.
        """
        super().__init__(alpha)
        assert len(benign_updates_this_round) > 2 * n_remove_malicious, "Must have more entries than malicious items being removed."

        # Organize updates per client into one single numpy matrix
        self.benign_clients = [np.zeros([*layer.shape, len(benign_updates_this_round)], layer.dtype)
                       for layer in benign_updates_this_round[0]]
        for client in range(0, len(benign_updates_this_round)):
            for layer in range(len(benign_updates_this_round[client])):
                self.benign_clients[layer][..., client] = benign_updates_this_round[client][layer]

        for layer in range(len(self.benign_clients)):
            sorted = np.sort(self.benign_clients[layer], -1)

            # Remove bottom and top values
            self.benign_clients[layer] = sorted[..., n_remove_malicious:-n_remove_malicious]

        # For now, take smallest of top and bottom values ... may backfire in future
        self.layer_val = [np.minimum(np.abs(layer[..., 0]), np.abs(layer[..., -1])) for layer in self.benign_clients]
        self.layer_val_tensor = [tf.convert_to_tensor(layer) for layer in self.layer_val]

    def loss_term(self, model):

        def loss(y_true, y_pred):
            weight_norm = tf.constant(0.0, dtype=tf.float32)
            layer_i = 0
            local_weights = model.layers
            for local_weight_layer in local_weights:
                w = local_weight_layer.weights
                if len(w) > 1:
                    global_layer_tensor_w = self.layer_val_tensor[layer_i]
                    # global_layer_tensor_b = tf.convert_to_tensor(global_weights[layer_i + 1])
                    delta_weight = w[0] - global_layer_tensor_w
                    # weight_norm += tf.nn.l2_loss(delta_weight)
                    weight_norm = tf.add(weight_norm, tf.nn.l2_loss(delta_weight))
                    layer_i += len(w)

            return weight_norm
        return loss

    def update_after_training(self, model):
        # Now we explicitly clip, not sure if this is needed as if we do not get selected the "second best" would be
        pass
