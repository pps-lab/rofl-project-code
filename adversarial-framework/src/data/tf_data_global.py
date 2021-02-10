from collections import defaultdict

import numpy as np
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

from src.data import image_augmentation
import logging

class GlobalDataset:
    """
    A GlobalDataset represents a dataset as a whole. It has two purposes.
    - Client datasets are derived from it
    - Our global dataset is used for evaluation of the global model. `x_test`, `y_test` and the aux sets

    """
    def __init__(self, x_test, y_test):
        self.x_test = x_test
        self.y_test = y_test
        self.x_train = []
        self.y_train = []

        self.x_aux_train, self.y_aux_train, self.mal_aux_labels_train = \
            [], [], []
        self.x_aux_test, self.y_aux_test, self.mal_aux_labels_test = \
            [], [], []

        self.test_generator = tf.data.Dataset.from_tensor_slices((x_test, y_test))

        self.aux_test_generator = None

    def get_dataset_for_client(self, client_id):
        raise Exception("Not implemented")

    def get_normal_and_aux_dataset_for_client(self, client_id, aux_sample_size, attack_objective):
        raise Exception("Not implemented")

    def get_test_batch(self, batch_size, max_num_batches=-1):
        """Creates one batch of test data.

        Yields:
            tuple of two: input data batch and corresponding labels
        """
        # count = min(int(self.x_test.shape[0] / batch_size), max_num_batches)
        # for bid in range(count):
        #     batch_x = self.x_test[bid * batch_size:(bid + 1) * batch_size]
        #     batch_y = self.y_test[bid * batch_size:(bid + 1) * batch_size]
        #
        #     yield batch_x, batch_y
        # bid = 0
        # Check here if non cifar?
        return self.test_generator.batch(batch_size) \
            .take(max_num_batches) \
            .map(image_augmentation.test_augment, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
            .prefetch(tf.data.experimental.AUTOTUNE)

# TODO 0: Do we want to use num_backdoor_tasks or aux_sample_size ?
    def build_global_aux(self, mal_clients, num_backdoor_tasks, attack_objective, aux_sample_size, augment_size):
        """ Select backdoor tasks """
        if np.count_nonzero(mal_clients) == 0:
            return # no aux
        assert np.count_nonzero(mal_clients) >= num_backdoor_tasks # assert we have less 'tasks' than clients

        data_x, data_y = self.x_train, self.y_train

        total_x_aux, total_y_aux, total_mal_aux_labels = [], [], []

        if aux_sample_size == -1:
            aux_sample_size = 10000000 # fix as we reformat this

        total_aux_count = 0
        num_tasks = 0
        for i in range(len(data_x)):
            if num_tasks >= num_backdoor_tasks:
                break
            if total_aux_count >= aux_sample_size:
                print(f"Hit limit of {total_aux_count}/{aux_sample_size} samples!")
                break
            if mal_clients[i]:
                x_train_me, y_train_me = data_x[i], data_y[i]

                # Pick attack samples
                inds = np.where(y_train_me == attack_objective[0])[0]  # Find all
                logging.debug(f"{i} Found {len(inds)} of class {attack_objective} to poison!")

                test_inds = np.ones(x_train_me.shape[0], dtype=bool)
                test_inds[inds] = False
                x_aux, y_aux = x_train_me[inds], y_train_me[inds]
                x_train_me, y_train_me = x_train_me[test_inds], y_train_me[test_inds]

                # randomly permute labels
                mal_labels = np.repeat(attack_objective[1], len(y_aux))

                current_aux_count = y_aux.size
                if total_aux_count + current_aux_count > aux_sample_size:
                    # constrain
                    current_aux_count = aux_sample_size - total_aux_count # how many we have left
                    x_aux = x_aux[:current_aux_count, :]
                    y_aux = y_aux[:current_aux_count]
                    mal_labels = mal_labels[:current_aux_count]

                total_x_aux.append(x_aux)
                total_y_aux.append(y_aux)
                total_mal_aux_labels.append(mal_labels)

                data_x[i], data_y[i] = x_train_me, y_train_me

                assert not np.any(
                    data_y[i] == attack_objective[0])  # assert data_y doesnt contain any attack label

                total_aux_count += current_aux_count
                num_tasks += 1

        # assert len(total_x_aux) == num_backdoor_tasks # not applicable with aux_sample_size
        self.x_aux_train = np.concatenate(total_x_aux)
        self.y_aux_train = np.concatenate(total_y_aux)
        self.mal_aux_labels_train = np.concatenate(total_mal_aux_labels).astype(np.uint8)

        # Assign train as test set for now ... ! Depends on how we want to implement the behavior
        self.x_aux_test = self.x_aux_train
        self.y_aux_test = self.y_aux_train
        self.mal_aux_labels_test = self.mal_aux_labels_train

        # self.build_aux_generator(augment_size)

        print(f"Got {len(self.x_aux_train)}/{aux_sample_size} samples for {num_backdoor_tasks} tasks!")

    # def build_aux_generator(self, augment_size):
    #     # self.aux_test_generator = tf.data.Dataset.from_tensor_slices((self.x_aux_test, self.y_aux_test))
    #     if augment_size == 0:
    #         self.aux_test_generator = ImageDataGenerator()
    #     else:
    #         self.aux_test_generator = ImageDataGenerator(
    #             # rotation_range=15,
    #             horizontal_flip=True,
    #             width_shift_range=0.1,
    #             height_shift_range=0.1
    #         )
    #     self.aux_test_generator.fit(self.x_aux_test)

    def get_aux_generator(self, batch_size, aux_size, augment_aux_set):
        if aux_size == 0:
            return tf.data.Dataset.from_tensor_slices((self.x_aux_test, self.mal_aux_labels_test)) \
                .batch(batch_size, drop_remainder=False) \
                .prefetch(tf.data.experimental.AUTOTUNE)

        test_dataset = tf.data.Dataset.from_tensor_slices((self.x_aux_test, self.mal_aux_labels_test)) \
            .repeat(aux_size) \
            .batch(batch_size, drop_remainder=False) \

        if augment_aux_set:
            return test_dataset\
                .map(image_augmentation.test_aux_augment, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
                .prefetch(tf.data.experimental.AUTOTUNE)
        else:
            return test_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    def get_full_dataset(self, size):
        x, y = np.concatenate(self.x_train), np.concatenate(self.y_train)
        perms = np.random.choice(x.shape[0], size, replace=False)
        x, y = x[perms, :], y[perms]
        return x, y


class IIDGlobalDataset(GlobalDataset):
    def __init__(self, x_train, y_train, num_clients, x_test, y_test):

        super().__init__(x_test, y_test)
        self.num_clients = num_clients

        x_train, y_train = self.shuffle(x_train, y_train)

        # Add to list
        for client_id in range(num_clients):
            data_samples = int(x_train.shape[0] / self.num_clients)
            inds = (client_id * data_samples, (client_id + 1) * data_samples)
            x, y = x_train[inds[0]:inds[1]], y_train[inds[0]:inds[1]]
            self.x_train.append(x)
            self.y_train.append(y)

    def shuffle(self, x, y):
        perms = np.random.permutation(x.shape[0])
        return x[perms, :], y[perms]

    def get_dataset_for_client(self, client_id):
        # dataset = tf.data.Dataset.from_tensor_slices((self.x_train[client_id], self.y_train[client_id]))
        # return dataset
        return self.x_train[client_id], self.y_train[client_id]

class NonIIDGlobalDataset(GlobalDataset):
    def __init__(self, x_train, y_train, x_test, y_test, num_clients):
        """Expects x_train to be a list, x_test one array"""

        super().__init__(x_test, y_test)

        self.x_train, self.y_train = x_train, y_train

    def shuffle(self):
        raise Exception("Shuffling is not supported on a non-IID dataset!")

    def get_dataset_for_client(self, client_id):
        return self.x_train[client_id], self.y_train[client_id]


class DirichletDistributionDivider():
    """Divides dataset according to dirichlet distribution"""

    def __init__(self, x_train, y_train, train_aux, test_aux, exclude_aux, num_clients):
        """`train_aux` and `test_aux` should be indices for the `train` arrays."""
        self.x_train = x_train
        self.y_train = y_train
        self.train_aux = train_aux
        self.test_aux = test_aux
        self.exclude_aux = exclude_aux
        self.num_clients = num_clients

    def build(self):
        alpha = 0.9
        cifar_classes = {}
        for ind, x in enumerate(self.y_train):
            label = x
            if self.exclude_aux and (ind in self.train_aux or ind in self.test_aux):
                continue
            if label in cifar_classes:
                cifar_classes[label].append(ind)
            else:
                cifar_classes[label] = [ind]
        class_size = len(cifar_classes[0])
        per_participant_list = defaultdict(list)
        no_classes = len(cifar_classes.keys())

        for n in range(no_classes):
            np.random.shuffle(cifar_classes[n])
            sampled_probabilities = class_size * np.random.dirichlet(
                np.array(self.num_clients * [alpha]))
            for user in range(self.num_clients):
                no_imgs = int(round(sampled_probabilities[user]))
                sampled_list = cifar_classes[n][:min(len(cifar_classes[n]), no_imgs)]
                per_participant_list[user].extend(sampled_list)
                cifar_classes[n] = cifar_classes[n][min(len(cifar_classes[n]), no_imgs):]

        per_participant_train_x = [self.x_train[ind] for _, ind in per_participant_list.items()]
        per_participant_train_y = [self.y_train[ind] for _, ind in per_participant_list.items()]

        for n in range(self.num_clients):
            perms = np.random.permutation(per_participant_train_x[n].shape[0])
            per_participant_train_x[n] = per_participant_train_x[n][perms, :]
            per_participant_train_y[n] = per_participant_train_y[n][perms]

        return (per_participant_train_x, per_participant_train_y)