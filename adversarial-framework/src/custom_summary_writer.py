import numpy as np
from tensorboardX import SummaryWriter
from os.path import join
from numpy.linalg import norm
from tensorflow.python.keras.layers.convolutional import Conv2D
from tensorflow.python.keras.layers.core import Dense
from scipy.linalg import eigh

from src.federated_averaging import FederatedAveraging
from src.util import power_iteration


class CustomSummaryWriter:

    def __init__(self, experiment_dir):
        self.events_dir = join(experiment_dir, 'events')
        self.norms_dir = join(experiment_dir, 'norms')

        self.writer = SummaryWriter(self.events_dir)

    def add_test_metric(self, test_accuracy, adv_success, step):
        """Adds performance metrics to tensorboard log files.

        Args:
            test_accuracy (float): Accuracy on the unseen set.
            adv_success (float): Adversarial success.
            step (int): Global step (round).
        """

        self.writer.add_scalar(f'evaluation/test_accuracy', test_accuracy, step)
        self.writer.add_scalar(f'evaluation/adv_success', adv_success, step)
        self.writer.flush()

    def add_honest_train_loss(self, selected_clients_list, step):
        train_loss = []
        for client in selected_clients_list:
            if not client.malicious:
                train_loss.append(client.train_loss.result())
        total_train_loss = np.mean(train_loss)
        self.writer.add_scalar(f'evaluation/honest_train_accuracy', total_train_loss, step)

    def add_adversary_count(self, num_adversaries_active, step):
        self.writer.add_scalar(f'adversary/count', num_adversaries_active, step)

    def analyze_weights(self, model, prev_weights, selected_clients, step, parameters_history=None, save_norms=False, save_histograms=False):
        """Analyzes model updates.

        Args:
            model (tf model): Current global model.
            selected_clients (list): List of Client objects.
            step (int): Global step (round).
            parameters_history (list): (Optional) List of weights from previous rounds.
        """
        # prev_weights = model.get_weights()
        benign_updates, mal_updates = [], []
        for client in selected_clients:
            if client.malicious:
                mal_updates.append(FederatedAveraging.compute_updates(prev_weights, client.weights))
            else:
                benign_updates.append(FederatedAveraging.compute_updates(prev_weights, client.weights))
        benign_update = FederatedAveraging.average_weights(benign_updates) if benign_updates != [] else [None]*len(prev_weights)
        mal_update = FederatedAveraging.average_weights(mal_updates) if mal_updates != [] else [None]*len(prev_weights)

        layer_names = ['Conv2D' if type(layer) == Conv2D else 'Dense'
                       for layer in model.layers if type(layer) in [Conv2D, Dense]]

        if len(layer_names) == 0:
            layer_names = ['Theta']

        printable_weights_index = [i for i in range(len(prev_weights)) if len(prev_weights[i].shape) > 1] # skip biases
        for i, layer in enumerate(printable_weights_index):
            for update, label in zip([mal_update[layer], benign_update[layer]], ['mal', 'benign']):
                if update is None:
                    continue

                suffix = f'norm_{label}_client_updates/l{layer}_{layer_names[i]}'
                # l2, l1 norm
                self.writer.add_scalar(f'l2_{suffix}', norm(update, axis=-1).mean(), step)
                self.writer.add_scalar(f'l1_{suffix}', norm(update, ord=1, axis=-1).mean(), step)
                # principle eig value
                if parameters_history is not None and len(parameters_history) > 1:
                    layer_history = [parameters[layer] for parameters in parameters_history]
                    princ_eig = self._principle_eigen_value(layer_history, layer_names[i])
                    self.writer.add_scalar(f'princ_eig_{suffix}', princ_eig, step)

            if save_histograms and len(mal_updates) > 0:
                mal_merged = np.concatenate([update[layer] for update in mal_updates])
                self.writer.add_histogram(f'histogram_mal/l{layer}_{layer_names[i]}', mal_merged.reshape(-1), step)
            if save_histograms and len(benign_updates) > 0:
                ben_merged = np.concatenate([update[layer] for update in benign_updates])
                self.writer.add_histogram(f'histogram_ben/l{layer}_{layer_names[i]}', ben_merged.reshape(-1), step)

                ben_for_layer = [update[layer] for update in benign_updates]
                means = []
                stds = []
                for x in ben_for_layer:
                    n, bins = np.histogram(x)
                    mids = 0.5 * (bins[1:] + bins[:-1])
                    probs = n / np.sum(n)

                    mean = np.sum(probs * mids)
                    sd = np.sqrt(np.sum(probs * (mids - mean) ** 2))
                    means.append(mean)
                    stds.append(sd)

                self.writer.add_histogram(f'histogram_ben_mean/l{layer}_{layer_names[i]}', np.array(means), step)
                self.writer.add_histogram(f'histogram_ben_std/l{layer}_{layer_names[i]}', np.array(stds), step)

        benign_norms_l2 = [norm(np.concatenate([np.reshape(b, [-1]) for b in bs], axis=0), axis=-1) for bs in benign_updates]
        benign_norms_l1 = [norm(np.concatenate([np.reshape(b, [-1]) for b in bs], axis=0), axis=-1, ord=1) for bs in
                           benign_updates]
        self.writer.add_scalar(f'l2_total/benign', np.mean(benign_norms_l2), step)
        self.writer.add_scalar(f'l1_total/benign', np.mean(benign_norms_l1), step)

        if len(mal_updates) > 0:
            mal_norms_l2 = [norm(np.concatenate([np.reshape(b, [-1]) for b in bs], axis=0), axis=-1) for bs in mal_updates]
            mal_norms_l1 = [norm(np.concatenate([np.reshape(b, [-1]) for b in bs], axis=0), axis=-1, ord=1) for bs in mal_updates]
            self.writer.add_scalar(f'l2_total/mal', np.mean(mal_norms_l2), step)
            self.writer.add_scalar(f'l1_total/mal', np.mean(mal_norms_l1), step)

            if save_norms:
                self.save_norms_log(step, np.array([benign_norms_l2, benign_norms_l1, mal_norms_l2, mal_norms_l1]))
        elif save_norms:
            self.save_norms_log(step, np.array([benign_norms_l2, benign_norms_l1]))

        self.writer.flush()

    def save_norms_log(self, step, array):
        np.save(join(self.norms_dir, f"round_{step}"), array)

    def write_hparams(self, hparams, metrics):
        self.writer.add_hparams(hparams, metrics)


    @staticmethod
    def _principle_eigen_value(layer_weights, layer_type=None):
        """Computes principle eigenvalue.

        Args:
            layer_weights (list): List of np.ndarray parameters.
            layer_type (str): List of layer names.

        Returns:
            float64: Principle eigenvalue.
        """
        layer_weights = np.stack([layer_params.reshape(-1) for layer_params in layer_weights], axis=0)  # NxM
        cov_matrix = np.cov(layer_weights.T)  # MxM
        # _w, _v = eigh(cov_matrix)  # princ w[-1]
        w, v = power_iteration(cov_matrix)
        return w  # largest eigenvalue
