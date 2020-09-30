from __future__ import absolute_import, division, print_function, unicode_literals

import json
import os
import time
from copy import deepcopy
from pathlib import Path
from threading import Thread
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf

from src.client import Client
from src.util import log_data, create_dropout_mask, aggregate_weights_masked
from src.tf_data import Dataset, ImageGeneratorDataset, GeneratorDataset
from src.tf_model import Model
from src.tf_data_global import IIDGlobalDataset, NonIIDGlobalDataset, DirichletDistributionDivider
from src.subspace.keras_ext.engine_training import ExtendedModel


class FederatedAveraging:
    """Implementation of federated averaging algorithm."""

    def __init__(self, config, models, run=None):
        self.config = config

        self.num_clients = config['num_clients']
        self.num_selected_clients = config['num_selected_clients']
        self.num_malicious_clients = config['num_malicious_clients']
        self.attack_frequency = config['attack_frequency']
        self.attack_type = config['attack_type']
        self.targeted_deterministic_attack_objective = config['targeted_deterministic_attack_objective']
        self.targeted_attack_objective = config['targeted_attack_objective']
        self.scale_attack = config['scale_attack']
        self.scale_attack_weight = config['scale_attack_weight']
        self.data_distribution = config['data_distribution']
        self.aux_samples = config['aux_samples']

        self.num_rounds = config['num_rounds']
        self.num_epochs = config['num_epochs']
        self.batch_size = config['batch_size']

        self.learning_rate = config['learning_rate']
        self.federated_dropout_rate = config['federated_dropout_rate']
        if config['global_learning_rate'] < 0:
            print("Using default global learning rate of n/m")
            self.global_learning_rate = self.num_clients / self.num_selected_clients
        else:
            self.global_learning_rate = config['global_learning_rate']

        self.print_every = config['print_every']

        self.model_name = config['model_name']

        self.workers = config['workers']

        self.experiment_name = config['experiment_name']

        self.clip = config['clip']

        self.model = models[0] # use first for me
        self.client_models = models
        self.global_weights = self.model.get_weights()

        self.experiment_root_dir = os.path.join(os.getcwd(), 'experiments')
        self.experiment_dir = os.path.join(self.experiment_root_dir, self.experiment_name)
        if run is not None:
            self.experiment_dir = os.path.join(self.experiment_dir, run)
        self.client_updates_dir = os.path.join(self.experiment_dir, 'updates')
        self.global_model_dir = os.path.join(self.experiment_dir, 'models')
        self.norms_dir = os.path.join(self.experiment_dir, 'norms')
        # self.clients_data = []
        self.malicious_clients = np.zeros(self.num_clients, dtype=bool)
        if self.num_malicious_clients > 0:
            self.malicious_clients[np.random.choice(self.num_clients, self.num_malicious_clients, replace=False)] = True

        self.global_dataset = self.build_dataset()
        # self.malicious_clients[np.random.choice(self.num_clients, self.num_malicious_clients, replace=False)] = True
        self.client_objs = []
        self.client_model = None
        self.client_config = {}

        self.writer = None
        self.keep_history = config['keep_history']
        self.parameters_history = [] if self.keep_history else None

        self.test_accuracy = tf.keras.metrics.Mean(name='test_accuracy')

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

        if self.config["save_norms"] and not os.path.isdir(self.norms_dir):
            os.mkdir(self.norms_dir)

        # remove everything for this directory
        for filename in Path(self.experiment_dir).glob('**/*'):
            if not os.path.isdir(str(filename)):
                os.remove(str(filename))

        with open(os.path.join(self.experiment_dir, 'config.json'), 'w') as fp:
            json.dump(self.config, fp, indent=4, sort_keys=True)

        from src.custom_summary_writer import CustomSummaryWriter
        self.writer = CustomSummaryWriter(self.experiment_dir)

    def init(self):
        """ Loads data, creates clients and client configuration."""
        self._init_log_directories()

        self.client_config = {
            'attack_type': self.config['attack_type'],
            'batch_size': self.config['batch_size'],
            'untargeted_after_training': self.config['untargeted_after_training'],
            'targeted_deterministic_attack_objective': self.config['targeted_deterministic_attack_objective'],
            'targeted_attack_objective': self.config['targeted_attack_objective'],
            'targeted_attack_benign_first': self.config['targeted_attack_benign_first'],
            'scale_attack': self.config['scale_attack'],
            'scale_attack_weight': self.config['scale_attack_weight'],
            'num_epochs': self.config['num_epochs'],
            'optimizer': self.config['optimizer'],
            'learning_rate': self.config['learning_rate'],
            'decay_steps': self.config['decay_steps'],
            'decay_rate': self.config['decay_rate'],
            'mal_learning_rate': self.config['mal_learning_rate'],
            'mal_decay_steps': self.config['mal_decay_steps'],
            'mal_decay_rate': self.config['mal_decay_rate'],
            'poison_samples': self.config['poison_samples'],
            'mal_num_batch': self.config['mal_num_batch'],
            'mal_step_learning_rate': self.config['mal_step_learning_rate'],
            'mal_num_epochs': self.config['mal_num_epochs'],
            'model_name': self.config['model_name'],
            'clip': self.config['clip'],
            'clip_probability': self.config['clip_probability'],
            'clip_l2': self.config['clip_l2'],
            'clip_layers': self.config['clip_layers'],
            'backdoor_stealth': self.config['backdoor_stealth'],
            'estimate_other_updates': self.config['estimate_other_updates'],
            'attack_after': self.config['attack_after'],
            'attack_stop_after': self.config['attack_stop_after'],
            'contamination_model': self.config['contamination_model'],
            'contamination_rate': self.config['contamination_rate'],
            'gaussian_noise': self.config['gaussian_noise'],
            'pgd': self.config['pgd'],
            'pgd_constraint': self.config['pgd_constraint'],
            'pgd_adaptive': self.config['pgd_adaptive'],
            'weight_regularization_alpha': self.config['weight_regularization_alpha'],
            'quantization': self.config['quantization'],
            'q_bits': self.config['q_bits'],
            'q_frac': self.config['q_frac'],
            'optimized_training': self.config['optimized_training']
        }

        self.build_clients(self.malicious_clients)
        #
        # if self.num_malicious_clients > 0:
        #     self.global_dataset.complete_aux()

        # self.client_model = Model.create_model(self.config["model_name"])
        # if isinstance(self.client_model, ExtendedModel):
        #     # Hack to prevent efficient aggregation
        #     # Set same randomly initialized weights once
        #     self.client_model.set_all_weights([self.model.weights[i].])

    def build_clients(self, mal_clients):
        if self.config['attack_type'] == 'min_loss' or self.config['attack_type'] == 'data_poison' or self.config['attack_type'] == 'backdoor_feature':
            for bid in range(self.num_clients):
                x, y = self.global_dataset.get_dataset_for_client(bid)
                ds = self.get_local_dataset(self.config['augment_data'], x, y, batch_size=self.batch_size)
                # dataset = self.global_dataset.get_dataset_for_client(bid)
                # ds = GeneratorDataset(dataset, self.batch_size)
                if mal_clients[bid]:
                    ds.x_aux, ds.y_aux, ds.mal_aux_labels = self.global_dataset.x_aux_train, \
                                                              self.global_dataset.y_aux_train, \
                                                              self.global_dataset.mal_aux_labels_train

                    if self.config['attacker_full_dataset']:
                        ds.x_train, ds.y_train = self.global_dataset.get_full_dataset()

                self.client_objs.append(Client(bid, self.client_config, ds, mal_clients[bid]))
        else:
            for bid in range(self.num_clients):
                x, y = self.global_dataset.get_dataset_for_client(bid)
                ds = self.get_local_dataset(self.config['augment_data'], x, y, batch_size=self.batch_size)
                self.client_objs.append(Client(bid, self.client_config, ds, mal_clients[bid]))

    @staticmethod
    def get_local_dataset(augment_data, x, y, batch_size):
        if augment_data:
            return ImageGeneratorDataset(x, y, batch_size=batch_size)
        else:
            return Dataset(x, y, batch_size=batch_size)

    def build_dataset(self):
        dataset = self.config['dataset']
        number_of_samples = self.config['number_of_samples']
        if dataset == 'mnist':
            if self.data_distribution == 'IID':
                (x_train, y_train), (x_test, y_test) = Dataset.get_mnist_dataset(number_of_samples)
                ds = IIDGlobalDataset(x_train, y_train, num_clients=self.num_clients, x_test=x_test, y_test=y_test)
            else:
                raise Exception("Not implemented yet")

            if self.attack_type == 'min_loss' or self.attack_type == 'data_poison':
                ds.build_global_aux(self.malicious_clients, self.config["backdoor_tasks"], self.config['backdoor_attack_objective'],
                                    self.config["aux_samples"], self.config['backdoor_feature_augment_times'])
        elif dataset == 'fmnist':
            if self.data_distribution == 'IID':
                (x_train, y_train), (x_test, y_test) = Dataset.get_fmnist_dataset(number_of_samples)
                ds = IIDGlobalDataset(x_train, y_train, num_clients=self.num_clients, x_test=x_test, y_test=y_test)
            else:
                raise Exception('Distribution not supported')

            if self.attack_type == 'min_loss' or self.attack_type == 'data_poison':
                ds.build_global_aux(self.malicious_clients, self.config["backdoor_tasks"],
                                    self.config['backdoor_attack_objective'],
                                    self.config["aux_samples"], self.config['backdoor_feature_augment_times'])

        elif dataset == 'femnist':
            if self.data_distribution == 'IID':
                (x_train, y_train), (x_test, y_test) = Dataset.get_emnist_dataset(-1,
                                                                                  self.num_clients)
                (x_train, y_train), (x_test, y_test) = (Dataset.keep_samples(np.concatenate(x_train), np.concatenate(y_train), number_of_samples),Dataset.keep_samples(np.concatenate(x_test), np.concatenate(y_test), number_of_samples))
                print(y_train.shape, y_test.shape)
                ds = IIDGlobalDataset(x_train, y_train, self.num_clients, x_test, y_test)
            else:
                (x_train, y_train), (x_test, y_test) = Dataset.get_emnist_dataset(number_of_samples,
                                                                                  self.num_clients)
                ds = NonIIDGlobalDataset(x_train, y_train, np.concatenate(x_test), np.concatenate(y_test),
                                         self.num_clients)

            if self.attack_type == 'min_loss' or self.attack_type == 'data_poison':
                ds.build_global_aux(self.malicious_clients, self.config["backdoor_tasks"],
                                    self.config['backdoor_attack_objective'],
                                    self.config["aux_samples"], self.config['backdoor_feature_augment_times'])

        elif dataset == 'cifar10':
            (x_train, y_train), (x_test, y_test) = Dataset.get_cifar10_dataset(number_of_samples)
            # # # Test visualize
            # # vis = np.array([[30696, 33105, 33615], [33907, 36848, 40713], [41706, 100, 121]])
            # vis = np.array([[2180, 2771, 3233], [4932, 6241, 6813]])
            #
            # fig, axes1 = plt.subplots(vis.shape[0], vis.shape[1], figsize=(3, 3))
            # for j in range(vis.shape[0]):
            #     for k in range(vis.shape[1]):
            #         i = vis[j, k]
            #         axes1[j][k].set_axis_off()
            #         axes1[j][k].imshow(x_train[i])
            #
            # plt.show()
            #
            # exit(0)
            if self.data_distribution == 'IID':
                ds = IIDGlobalDataset(x_train, y_train, num_clients=self.num_clients, x_test=x_test, y_test=y_test)
                if self.attack_type == 'min_loss' or self.attack_type == 'data_poison':
                    if self.config['backdoor_feature_aux_train'] != [] and self.config['backdoor_feature_aux_test']:
                        self.create_backdoor_feature_aux(ds, x_train, y_train)
                    else:
                        ds.build_global_aux(self.malicious_clients, self.config["backdoor_tasks"],
                                            self.config['backdoor_attack_objective'],
                                            self.config["aux_samples"], self.config['backdoor_feature_augment_times'])
            else:
                (x_train_dist, y_train_dist) = \
                    DirichletDistributionDivider(x_train, y_train, self.config['backdoor_feature_aux_train'],
                                                 self.config['backdoor_feature_aux_test'],
                                                 self.num_clients).build()
                ds = NonIIDGlobalDataset(x_train_dist, y_train_dist, x_test, y_test, num_clients=self.num_clients)
                if self.attack_type == 'min_loss' or self.attack_type == 'data_poison':
                    if self.config['backdoor_feature_aux_train'] != [] and self.config['backdoor_feature_aux_test']:
                        self.create_backdoor_feature_aux(ds, x_train, y_train)
                    else:
                        ds.build_global_aux(self.malicious_clients, self.config["backdoor_tasks"],
                                            self.config['backdoor_attack_objective'],
                                            self.config["aux_samples"], self.config['backdoor_feature_augment_times'])
        else:
            raise Exception('Selected dataset with distribution not supported')

        return ds

    def create_backdoor_feature_aux(self, ds, x_train, y_train):
        (ds.x_aux_train, ds.y_aux_train), (ds.x_aux_test, ds.y_aux_test) = \
            (x_train[np.array(self.config['backdoor_feature_aux_train'])],
             y_train[np.array(self.config['backdoor_feature_aux_train'])]), \
            (x_train[np.array(self.config['backdoor_feature_aux_test'])],
             y_train[np.array(self.config['backdoor_feature_aux_test'])])
        ds.mal_aux_labels_train = np.repeat(self.config['backdoor_feature_target'],
                                            ds.y_aux_train.shape)
        ds.mal_aux_labels_test = np.repeat(self.config['backdoor_feature_target'], ds.y_aux_test.shape)

        if self.config['backdoor_feature_benign_regular']:
            extra_train_x, extra_train_y = x_train[np.array(self.config['backdoor_feature_benign_regular'])], \
                                           y_train[np.array(self.config['backdoor_feature_benign_regular'])]
            ds.x_aux_train = np.concatenate([ds.x_aux_train, extra_train_x])
            ds.y_aux_train = np.concatenate([ds.y_aux_train, extra_train_y])
            ds.mal_aux_labels_train = np.concatenate([ds.mal_aux_labels_train, extra_train_y])

        if self.config['backdoor_feature_remove_malicious']:
            np.delete(x_train, self.config['backdoor_feature_aux_train'], axis=0)
            np.delete(y_train, self.config['backdoor_feature_aux_train'], axis=0)
            np.delete(x_train, self.config['backdoor_feature_aux_test'], axis=0)
            np.delete(y_train, self.config['backdoor_feature_aux_test'], axis=0)

        ds.build_aux_generator(self.config['backdoor_feature_augment_times'])

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
        if self.federated_dropout_rate == 1.0:
            return None, {i: self.model.get_weights() for i in selected_clients}

        # create dropout mask for each client
        if self.config['federated_dropout_nonoverlap']:
            client_dropout_mask = create_dropout_mask(self.model, self.federated_dropout_rate,
                                                      self.config['federated_dropout_all_parameters'],
                                                      n_clients=len(selected_clients))
        else:
            client_dropout_mask = []
            for _ in selected_clients:
                client_dropout_mask.append(create_dropout_mask(self.model, self.federated_dropout_rate,
                                                               self.config['federated_dropout_all_parameters'])[0])

        # apply dropout mask
        weights_list = {}
        for i, dropout_mask in enumerate(client_dropout_mask):
            self.client_objs[selected_clients[i]].set_dropout_mask(dropout_mask)

            model_weights = deepcopy(self.model.get_weights())
            if not self.config['federated_dropout_randommask']:
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

        print("Starting training...")
        for round in range(1, self.num_rounds + 1):

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

            if self.workers > 1:
                # raise Exception("This setup causes memory leaks! Tensorflow already parallelizes ops!")
                for i in range(0, self.num_selected_clients, self.workers):
                    workers = self.workers
                    if i > self.num_selected_clients - self.workers:
                        workers = self.num_selected_clients - i

                    # prepare clients
                    for w in range(workers):
                        client_id = selected_clients[i + w]
                        self.client_objs[client_id].set_weights(weights_list[client_id])
                        self.client_objs[client_id].set_model(self.client_models[w])

                    threads = [Thread(target=self.client_objs[selected_clients[i + w]].train, args=[round]) for w in range(workers)]

                    for thread in threads:
                        thread.start()

                    for thread in threads:
                        thread.join()

                    for w in range(workers):
                        client_id = selected_clients[i + w]
                        self.client_objs[client_id].set_model(None)
            else:
                for i in selected_clients:
                    self.client_objs[i].set_weights(weights_list[i])
                    self.client_objs[i].set_model(self.model)
                    # self.client_objs[i].optimizer = central_optimizer
                    self.client_objs[i].train(round)
                    self.client_objs[i].set_model(None)

            if self.config['save_updates']:
                for i in selected_clients:
                    self.save_client_updates(self.client_objs[i].id, self.client_objs[i].malicious, round,
                                             self.client_objs[i].weights, weights_list[i])

            num_adversaries = np.count_nonzero([self.malicious_clients[i] for i in selected_clients])
            selected_clients_list = [self.client_objs[i] for i in selected_clients]
            if client_dropout_masks is not None:
                # Federated Dropout
                weights = aggregate_weights_masked(self.global_weights, self.global_learning_rate, self.num_clients, self.federated_dropout_rate, client_dropout_masks, [client.weights for client in selected_clients_list])
                # weights = self.aggregate_weights([client.weights for client in selected_clients_list])

            elif self.config['ignore_malicious_update']:
                # Ignore malicious updates
                weights = self.aggregate_weights([client.weights for client in selected_clients_list if not client.malicious])
            else:
                weights = self.aggregate_weights([client.weights for client in selected_clients_list])


            if self.keep_history:
                self.parameters_history.append(deepcopy(weights))

            if round % self.print_every == 0:
                if Model.model_supports_weight_analysis(self.model_name):
                    self.writer.analyze_weights(self.model, self.global_weights, selected_clients_list, round, self.parameters_history, self.config['save_norms'], self.config['save_weight_distributions'])

                self.model.set_weights(weights)
                self.global_weights = weights

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

            if round in self.config['save_model_at']:
                self.save_model(round)

            for client in self.client_objs:
                client.weights = None # Release

        log_data(self.experiment_dir, rounds, accuracies, adv_success_list)

    def aggregate_weights(self, client_weight_list):
        """Procedure for merging client weights together with `global_learning_rate`."""
        current_weights = self.global_weights
        new_weights = deepcopy(current_weights)
        # return new_weights
        update_coefficient = self.global_learning_rate / self.num_clients

        print(f"Update {update_coefficient} {self.global_learning_rate}/{self.num_clients}, {self.num_selected_clients} {self.config['scale_attack_weight']}")

        for client in range(0, len(client_weight_list)):
            for layer in range(len(client_weight_list[client])):
                new_weights[layer] = new_weights[layer] + \
                                     update_coefficient * (client_weight_list[client][layer] - current_weights[layer])

        return new_weights

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
        # loss = loss_object(y_true=batch_y, y_pred=prediction_tensor)
        # print(f"P {loss}")
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

        for batch_x, batch_y in self.global_dataset.get_test_batch(self.config['batch_size'], self.config['num_test_batches']):
            self.optimized_evaluate(batch_x, batch_y)

        test_accuracy = self.test_accuracy.result().numpy()
        # print(test_accuracy.nump)

        # calculating adv success
        if self.num_malicious_clients == 0:
            adv_success = 0
        elif self.attack_type == 'untargeted':
            adv_success = 1 - test_accuracy
        elif self.attack_type == 'targeted_deterministic':
            inds = self.global_dataset.y_test != self.targeted_deterministic_attack_objective
            # adv_success = predictions[inds, self.targeted_deterministic_attack_objective].mean()
            raise NotImplementedError("Currently not implemented")
        elif self.attack_type == 'targeted':
            inds_gt = self.global_dataset.y_test == self.targeted_attack_objective[0]
            # inds_pred = y_ == self.targeted_attack_objective[1]
            # inds = np.logical_and(inds_gt, inds_pred)
            inds = inds_gt
            raise NotImplementedError("Currently not implemented")
            adv_success = predictions[inds, self.targeted_attack_objective[1]].mean()
        elif self.attack_type == 'min_loss' or self.attack_type == 'data_poison':
            all_adv_success = []
            batches = 0
            amount_images = max(self.config['backdoor_feature_augment_times'], self.global_dataset.x_aux_test.shape[0])
            batch_size = min(self.global_dataset.x_aux_test.shape[0], self.config['batch_size'])
            total_batches = int(amount_images / batch_size) # handle case ?
            # for batch_x, batch_y in self.global_dataset.aux_test_generator.batch(self.config['batch_size']).prefetch(tf.data.experimental.AUTOTUNE):
            # for batch_x, batch_y in self.global_dataset.aux_test_generator.batch(self.config['batch_size']).prefetch(tf.data.experimental.AUTOTUNE):
            for batch_x, batch_y in self.global_dataset.aux_test_generator.flow(self.global_dataset.x_aux_test,
                                                                                self.global_dataset.mal_aux_labels_test,
                                                                                self.config['batch_size']):
                preds = self.model(batch_x, training=False).numpy().argmax(axis=1)
                pred_inds = preds == batch_y
                # print(f"Correct: {self.global_dataset.y_aux_test[pred_inds]} -> {preds[pred_inds]}")
                adv_success = np.mean(pred_inds)
                all_adv_success.append(adv_success)
                batches += 1
                if batches > total_batches:
                    break # manually

            adv_success = np.mean(all_adv_success)
        else:
            raise Exception('Type not supported')

        return test_accuracy, adv_success

    def save_model(self, round):
        path = os.path.join(self.global_model_dir, f'model_{round}.h5')
        print(f"Saving model at {path}")
        self.model.save(path)

    def write_hparams(self, hparams, metrics):
        self.writer.write_hparams(hparams, metrics)