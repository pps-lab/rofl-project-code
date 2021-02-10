from __future__ import absolute_import, division, print_function, unicode_literals

import json
import os
import time
import logging
from copy import deepcopy
from pathlib import Path
from threading import Thread

import numpy as np
import tensorflow as tf

import src.config as config
from src.attack_dataset_config import AttackDatasetConfig
from src.aggregation import aggregators
from src.data import data_loader
from src.client_attacks import Attack
from src.client import Client
from src.util import log_data, create_dropout_mask, aggregate_weights_masked, flatten
from src.data.tf_data import Dataset, ImageGeneratorDataset, GeneratorDataset
from src.tf_model import Model
from src.data.tf_data_global import IIDGlobalDataset, NonIIDGlobalDataset, DirichletDistributionDivider


class FederatedAveraging:
    """Implementation of federated averaging algorithm."""
    federated_dropout: config.definitions.FederatedDropout

    def __init__(self, config, models, config_path):
        """

        :type config: config_cli.Config
        """
        self.config = config

        self.num_clients = config.environment.num_clients
        self.num_selected_clients = config.environment.num_selected_clients
        self.num_malicious_clients = config.environment.num_malicious_clients
        self.attack_frequency = config.environment.attack_frequency
        self.attack_type = Attack(config.client.malicious.attack_type) \
            if config.client.malicious is not None else None

        self.num_rounds = config.server.num_rounds
        self.batch_size = config.client.benign_training.batch_size

        self.federated_dropout = config.server.federated_dropout

        self.attack_dataset = AttackDatasetConfig(**self.config.client.malicious.backdoor) \
            if self.config.client.malicious is not None and self.config.client.malicious.backdoor is not None else None

        self.print_every = config.environment.print_every

        self.model_name = config.client.model_name

        self.experiment_name = config.environment.experiment_name

        self.model = models[0] # use first for me
        self.client_models = models
        self.global_weights = self.model.get_weights()

        if config.environment.use_config_dir:
            self.experiment_dir = os.path.dirname(config_path)
            self.experiment_root_dir = os.path.dirname(self.experiment_dir)
        else:
            self.experiment_root_dir = os.path.join(os.getcwd(), 'experiments')
            self.experiment_dir = os.path.join(self.experiment_root_dir, self.experiment_name)

        self.client_updates_dir = os.path.join(self.experiment_dir, 'updates')
        self.global_model_dir = os.path.join(self.experiment_dir, 'models')
        self.norms_dir = os.path.join(self.experiment_dir, 'norms')
        # self.clients_data = []
        self.malicious_clients = np.zeros(self.num_clients, dtype=bool)
        if self.num_malicious_clients > 0:

            if config.environment.malicious_client_indices is not None:
                malicious_indices = config.environment.malicious_client_indices
            else:
                malicious_indices = np.random.choice(self.num_clients, self.num_malicious_clients, replace=False)

            assert len(malicious_indices) == self.num_malicious_clients, \
                "Malicious indices must equal total number of malicious clients!"
            self.malicious_clients[malicious_indices] = True

        self.global_dataset = self.build_dataset()
        # self.malicious_clients[np.random.choice(self.num_clients, self.num_malicious_clients, replace=False)] = True
        self.client_objs = []
        self.client_model = None
        self.client_config = {}

        self.writer = None
        self.keep_history = config.environment.save_history
        self.parameters_history = [] if self.keep_history else None

        self.test_accuracy = tf.keras.metrics.Mean(name='test_accuracy')

        self.aggregator = aggregators.build_aggregator(config)

    def _init_log_directories(self):
        """Initializes directories in which log files are stored"""
        if not os.path.isdir(self.experiment_root_dir):
            os.mkdir(self.experiment_root_dir)

        if not os.path.isdir(self.experiment_dir):
            os.mkdir(self.experiment_dir)

        if not os.path.isdir(self.client_updates_dir):
            os.mkdir(self.client_updates_dir)

        if not os.path.isdir(self.global_model_dir):
            os.mkdir(self.global_model_dir)

        if self.config.environment.save_norms and not os.path.isdir(self.norms_dir):
            os.mkdir(self.norms_dir)

        # remove everything for this directory, if we do not have set directory
        if not self.config.environment.use_config_dir:
            for filename in Path(self.experiment_dir).glob('**/*'):
                if not os.path.isdir(str(filename)):
                    os.remove(str(filename))

        # with open(os.path.join(self.experiment_dir, 'config.json'), 'w') as fp:
        #     self.config.to_yaml()

        from src.custom_summary_writer import CustomSummaryWriter
        self.writer = CustomSummaryWriter(self.experiment_dir)

    def init(self):
        """ Loads data, creates clients and client configuration."""
        self._init_log_directories()

        # self.client_config = {
        #     'attack': self.config['attack'],
        #     'attack_type': self.attack_type,
        #     'batch_size': self.config['batch_size'],
        #     'untargeted_after_training': self.config['untargeted_after_training'],
        #     'targeted_deterministic_attack_objective': self.config['targeted_deterministic_attack_objective'],
        #     'targeted_attack_objective': self.config['targeted_attack_objective'],
        #     'targeted_attack_benign_first': self.config['targeted_attack_benign_first'],
        #     'scale_attack': self.config['scale_attack'],
        #     'scale_attack_weight': self.config['scale_attack_weight'],
        #     'aggregator': self.config['aggregator'],
        #     'trimmed_mean_beta': self.config['trimmed_mean_beta'],
        #     'num_epochs': self.config['num_epochs'],
        #     'optimizer': self.config['optimizer'],
        #     'learning_rate': self.config['learning_rate'],
        #     'lr_decay': self.config['lr_decay'],
        #     'decay_steps': self.config['decay_steps'],
        #     'decay_rate': self.config['decay_rate'],
        #     'decay_boundaries': self.config['decay_boundaries'],
        #     'decay_values': self.config['decay_values'],
        #     'mal_learning_rate': self.config['mal_learning_rate'],
        #     'mal_decay_steps': self.config['mal_decay_steps'],
        #     'mal_decay_rate': self.config['mal_decay_rate'],
        #     'poison_samples': self.config['poison_samples'],
        #     'mal_num_batch': self.config['mal_num_batch'],
        #     'mal_step_learning_rate': self.config['mal_step_learning_rate'],
        #     'mal_num_epochs': self.config['mal_num_epochs'],
        #     'mal_num_epochs_max': self.config['mal_num_epochs_max'],
        #     'mal_target_loss': self.config['mal_target_loss'],
        #     'model_name': self.config['model_name'],
        #     'clip': self.config['clip'],
        #     'clip_probability': self.config['clip_probability'],
        #     'clip_l2': self.config['clip_l2'],
        #     'clip_layers': self.config['clip_layers'],
        #     'backdoor_stealth': self.config['backdoor_stealth'],
        #     'estimate_other_updates': self.config['estimate_other_updates'],
        #     'attack_after': self.config['attack_after'],
        #     'attack_stop_after': self.config['attack_stop_after'],
        #     'contamination_model': self.config['contamination_model'],
        #     'contamination_rate': self.config['contamination_rate'],
        #     'gaussian_noise': self.config['gaussian_noise'],
        #     'pgd': self.config['pgd'],
        #     'pgd_constraint': self.config['pgd_constraint'],
        #     'pgd_clip_frequency': self.config['pgd_clip_frequency'],
        #     'pgd_adaptive': self.config['pgd_adaptive'],
        #     'weight_regularization_alpha': self.config['weight_regularization_alpha'],
        #     'quantization': self.config['quantization'],
        #     'q_bits': self.config['q_bits'],
        #     'q_frac': self.config['q_frac'],
        #     'optimized_training': self.config['optimized_training']
        # }
        self.client_config = self.config.client

        self.build_clients(self.malicious_clients)

    def build_clients(self, mal_clients):
        if self.attack_type == Attack.BACKDOOR:
            for bid in range(self.num_clients):
                x, y = self.global_dataset.get_dataset_for_client(bid)
                if self.config.environment.attacker_full_dataset:
                    x, y = self.global_dataset.get_full_dataset(x.shape[0] * 20)

                ds = self.get_local_dataset(self.attack_dataset.augment_data, x, y, batch_size=self.batch_size)
                # dataset = self.global_dataset.get_dataset_for_client(bid)
                # ds = GeneratorDataset(dataset, self.batch_size)
                if mal_clients[bid]:
                    ds.x_aux, ds.y_aux, ds.mal_aux_labels = self.global_dataset.x_aux_train, \
                                                              self.global_dataset.y_aux_train, \
                                                              self.global_dataset.mal_aux_labels_train
                    ds.x_aux_test, ds.mal_aux_labels_test = self.global_dataset.x_aux_test, \
                                                            self.global_dataset.mal_aux_labels_test


                self.client_objs.append(Client(bid, self.client_config, ds, mal_clients[bid]))
        else:
            for bid in range(self.num_clients):
                x, y = self.global_dataset.get_dataset_for_client(bid)
                ds = self.get_local_dataset(self.config.dataset.augment_data, x, y, batch_size=self.batch_size)
                self.client_objs.append(Client(bid, self.client_config, ds, mal_clients[bid]))

    @staticmethod
    def get_local_dataset(augment_data, x, y, batch_size):
        if augment_data:
            return ImageGeneratorDataset(x, y, batch_size=batch_size)
        else:
            return Dataset(x, y, batch_size=batch_size)


    def build_dataset(self):
        return data_loader.load_global_dataset(self.config, self.malicious_clients, self.attack_dataset)


    @staticmethod
    def compute_updates(prev_weights, new_weights):
        """Compute difference between two model weights.

        Args:
            prev_weights (list): Parameters from previous iteration.
            new_weights (list): New weights.

        Returns:
            list: List of gradients.
        """
        return [new_weights[i] - prev_weights[i] for i in range(len(prev_weights))]

    def save_client_updates(self, client_id, malicious, round, new_weights, prev_global_model_weights):
        """Saves client updates into `self.client_updates_dir` directory."""
        if type(malicious) is not str:
            malicious = 'm' if malicious else 'b'

        delta_weights = self.compute_updates(prev_global_model_weights, new_weights)
        file_name = '%i_%s_%i' % (client_id, malicious, round)
        outfile = os.path.join(self.client_updates_dir, file_name)
        np.save(outfile, delta_weights)

    def _create_weights_list(self, selected_clients):
        """Creates dictionary (client weights for each selected client).
        Additionally, it sets dropout masks if federated_dropout is < 1.0.

        Args:
            selected_clients (np.ndarray): Randomly selected clients without replacement.

        Returns:
            dict: Mappings of client id -> model parameters.
        """
        if self.federated_dropout is None:
            return None, {i: self.model.get_weights() for i in selected_clients}

        # create dropout mask for each client
        if self.federated_dropout.nonoverlap:
            client_dropout_mask = create_dropout_mask(self.model, self.federated_dropout.rate,
                                                      self.federated_dropout.all_parameters,
                                                      n_clients=len(selected_clients))
        else:
            client_dropout_mask = []
            for _ in selected_clients:
                client_dropout_mask.append(create_dropout_mask(self.model, self.federated_dropout.rate,
                                                               self.federated_dropout.all_parameters)[0])

        # apply dropout mask
        weights_list = {}
        for i, dropout_mask in enumerate(client_dropout_mask):
            self.client_objs[selected_clients[i]].set_dropout_mask(dropout_mask)

            model_weights = deepcopy(self.model.get_weights())
            if not self.federated_dropout.randommask:
                for l in range(len(model_weights)):
                    model_weights[l] = model_weights[l]*dropout_mask[l]
            weights_list[selected_clients[i]] = model_weights

        return client_dropout_mask, weights_list

    def fit(self):
        """Trains the global model."""

        # central_optimizer = Model.create_optimizer(self.config['optimizer'], self.config['learning_rate'],
        #                                         self.config['decay_steps'], self.config['decay_rate']) # REMOVE THIS

        accuracies, rounds, adv_success_list = [], [], []


        # loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        #     from_logits=False)  # Our model has a softmax layer!
        # self.model.compile(
        #     loss=loss_object,
        #     optimizer=tf.keras.optimizers.Adam(0.001),
        #     metrics=['accuracy']
        # )

        logging.info("Starting training...")
        test_accuracy, adv_success = self.evaluate()
        print('round=', 0, '\ttest_accuracy=', test_accuracy, '\tadv_success=', adv_success, flush=True)

        import os
        import psutil

        for round in range(1, self.num_rounds + 1):

            process = psutil.Process(os.getpid())
            logging.debug("Memory info: " + str(process.memory_info().rss))  # in bytes

            start_time = time.time()

            if self.attack_frequency is None:
                selected_clients = np.random.choice(self.num_clients, self.num_selected_clients, replace=False)
            else:
                indexes = np.array([[i, self.client_objs[i].malicious] for i in range(len(self.client_objs))])
                np.random.shuffle(indexes)

                assert len(indexes[indexes[:, 1] == True]) > 0, "There are 0 malicious attackers."

                if round % (1 / self.attack_frequency) == 0:
                    honest = indexes[indexes[:, 1] == False][:self.num_selected_clients - 1, 0]
                    malicious = indexes[indexes[:, 1] == True][0][0:1]
                    selected_clients = np.concatenate([malicious, honest])
                else:
                    honest = indexes[indexes[:, 1] == False][:self.num_selected_clients, 0]
                    selected_clients = honest
                    assert len(selected_clients) == self.num_selected_clients, "There must be enough non-malicious clients to select."

            client_dropout_masks, weights_list = self._create_weights_list(selected_clients)

            # If attacker has full knowledge of a round
            intermediate_benign_client_weights = [] if self.config.environment.attacker_full_knowledge else None

            #################
            # TRAINING LOOP #
            #################
            for i in (c for c in selected_clients if not self.client_objs[c].malicious): # Benign
                self.client_objs[i].set_weights(weights_list[i])
                self.client_objs[i].set_model(self.model)
                self.client_objs[i].train(round)
                self.client_objs[i].set_model(None)
                if self.config.environment.attacker_full_knowledge:
                    intermediate_benign_client_weights.append(
                        FederatedAveraging.compute_updates(self.client_objs[i].weights, weights_list[i])
                    )

            for i in (c for c in selected_clients if self.client_objs[c].malicious): # Malicious
                self.client_objs[i].set_weights(weights_list[i])
                self.client_objs[i].set_model(self.model)
                self.client_objs[i].set_benign_updates_this_round(intermediate_benign_client_weights)
                self.client_objs[i].train(round)
                self.client_objs[i].set_model(None)

            if self.config.environment.save_updates:
                for i in selected_clients:
                    self.save_client_updates(self.client_objs[i].id, self.client_objs[i].malicious, round,
                                             self.client_objs[i].weights, weights_list[i])

            num_adversaries = np.count_nonzero([self.malicious_clients[i] for i in selected_clients])
            selected_clients_list = [self.client_objs[i] for i in selected_clients]
            if client_dropout_masks is not None:
                # Federated Dropout
                weights = aggregate_weights_masked(self.global_weights, self.config.server.global_learning_rate, self.num_clients, self.federated_dropout.rate, client_dropout_masks, [client.weights for client in selected_clients_list])
                # weights = self.aggregate_weights([client.weights for client in selected_clients_list])

            elif self.config.environment.ignore_malicious_update:
                # Ignore malicious updates
                temp_weights = [client.weights for client in selected_clients_list if not client.malicious]
                weights = self.aggregator.aggregate(self.global_weights, temp_weights)
            else:
                temp_weights = [client.weights for client in selected_clients_list]
                weights = self.aggregator.aggregate(self.global_weights, temp_weights)

            if self.config.server.gaussian_noise > 0.0:
                logging.debug(f"Adding noise to aggregated model {self.config.server.gaussian_noise}")
                weights = [layer + self.noise_with_layer(layer) for layer in weights]

            if self.keep_history:
                self.parameters_history.append(deepcopy(weights))

            if round % self.print_every == 0:
                self.model.set_weights(weights)
                self.global_weights = weights

                if Model.model_supports_weight_analysis(self.model_name):
                    self.writer.analyze_weights(self.model, self.global_weights, selected_clients_list, round, self.parameters_history,
                                                self.config.environment.save_norms, self.config.environment.save_weight_distributions)


                test_accuracy, adv_success = self.evaluate()
                duration = time.time() - start_time
                self.writer.add_test_metric(test_accuracy, adv_success, round)
                self.writer.add_honest_train_loss(selected_clients_list, round)
                self.writer.add_adversary_count(num_adversaries, round)

                accuracies.append(test_accuracy)
                adv_success_list.append(adv_success)
                rounds.append(round)
                print('round=', round, '\ttest_accuracy=', test_accuracy, '\tadv_success=', adv_success, '\tduration=', duration, flush=True)
            else:
                self.model.set_weights(weights)
                self.global_weights = weights

            if round in self.config.environment.save_model_at:
                self.save_model(round)

            for client in self.client_objs:
                client.weights = None # Release

        log_data(self.experiment_dir, rounds, accuracies, adv_success_list)
        self.log_hparams(rounds, accuracies, adv_success_list)

    def noise_with_layer(self, w):
        sigma = self.config.server.gaussian_noise
        gauss = np.random.normal(0, sigma, w.shape)
        gauss = gauss.reshape(w.shape).astype(w.dtype)
        return gauss

    @staticmethod
    def average_weights(client_weight_list):
        """Procedure for averaging client weights"""
        new_weights = deepcopy(client_weight_list[0])
        # return new_weights
        for client in range(1, len(client_weight_list)):
            for layer in range(len(client_weight_list[client])):
                new_weights[layer] = new_weights[layer] + client_weight_list[client][layer]

        for layer in range(len(new_weights)):
            new_weights[layer] = new_weights[layer] / len(client_weight_list)
        return new_weights

    @tf.function
    def optimized_evaluate(self, batch_x, batch_y):
        prediction_tensor = self.model(batch_x, training=False)
        # loss = tf.keras.losses.SparseCategoricalCrossentropy(
        #     from_logits=False)(y_true=batch_y, y_pred=prediction_tensor)
        # tf.print(loss)
        prediction = prediction_tensor
        y_ = tf.cast(tf.argmax(prediction, axis=1), tf.uint8)
        test_accuracy_batch = tf.equal(y_, batch_y)
        self.test_accuracy(tf.reduce_mean(tf.cast(test_accuracy_batch, tf.float32)))

    def evaluate(self):
        """Evaluates model performances; accuracy on test set and adversarial success.

        Returns:
            tuple of two floats: test accuracy, adversarial success
        """
        # return 0, 0
        # Batched because of memory issues
        # test_accuracies = []
        # predictions = []

        # loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        self.test_accuracy.reset_states()

        for batch_x, batch_y in self.global_dataset.get_test_batch(
                self.config.client.benign_training.batch_size, self.config.server.num_test_batches):
            self.optimized_evaluate(batch_x, batch_y)

        test_accuracy = self.test_accuracy.result().numpy()
        # print(test_accuracy.nump)

        # self.test_get_correct_indices()

        # calculating adv success
        if self.num_malicious_clients == 0:
            adv_success = 0
        elif self.attack_type == Attack.UNTARGETED or self.attack_type == Attack.DEVIATE_MAX_NORM:
            adv_success = 1 - test_accuracy
        elif self.attack_type == Attack.BACKDOOR:
            all_adv_success = []
            batches = 0
            attack_config: AttackDatasetConfig = self.attack_dataset
            amount_images = max(attack_config.augment_times, self.global_dataset.x_aux_test.shape[0])
            batch_size = min(self.global_dataset.x_aux_test.shape[0], self.config.client.benign_training.batch_size)
            total_batches = int(amount_images / batch_size) # handle case ?

            for batch_x, batch_y in self.global_dataset.get_aux_generator(self.config.client.benign_training.batch_size,
                                                                          attack_config.augment_times,
                                                                          attack_config.augment_data):
                preds = self.model(batch_x, training=False).numpy().argmax(axis=1)
                pred_inds = preds == batch_y
                if self.config.environment.print_backdoor_eval:
                    logging.info(f"Backdoor predictions: {preds}")

                # This may break on large test sets
                # adv_success = np.mean(pred_inds)
                all_adv_success.append(pred_inds)
                batches += 1
                if batches > total_batches:
                    break # manually

            adv_success = np.mean(np.concatenate(all_adv_success))
        else:
            raise Exception('Type not supported')

        return test_accuracy, adv_success

    def test_get_correct_indices(self):
        """Debug helper"""
        from src.backdoor.edge_case_attack import EuropeanSevenEdgeCase
        # (batch_x, batch_y), (_, _) = EuropeanSevenEdgeCase().load()
        (_, _), (batch_x, batch_y) = EuropeanSevenEdgeCase().load()
        as_7s = np.repeat(7, batch_y.shape)
        preds = self.model(batch_x, training=False).numpy().argmax(axis=1)
        pred_inds = preds == as_7s
        print(np.where(preds == as_7s))
        # print(f"Correct: {self.global_dataset.y_aux_test[pred_inds]} -> {preds[pred_inds]}")

    def save_model(self, round):
        path = os.path.join(self.global_model_dir, f'model_{round}.h5')
        print(f"Saving model at {path}")
        self.model.save(path)

    def write_hparams(self, hparams, metrics):
        self.writer.write_hparams(hparams, metrics)

    def log_hparams(self, rounds, accuracies, adv_successes):
        if self.config.hyperparameters is None:
            return

        # for now only log last round's values

        METRIC_ACCURACY = 'evaluation/test_accuracy'
        METRIC_ADV_SUCCESS = 'evaluation/adv_success'

        hparams_dict = flatten(self.config.hyperparameters.args)
        metrics = {
            METRIC_ACCURACY: accuracies[-1],
            METRIC_ADV_SUCCESS: adv_successes[-1]
        }

        self.writer.write_hparams(hparams_dict, metrics)

