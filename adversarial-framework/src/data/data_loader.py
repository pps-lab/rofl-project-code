from src.attack_dataset_config import AttackDatasetConfig
from src.backdoor.edge_case_attack import EdgeCaseAttack
from src.client_attacks import Attack
from src.data.tf_data import Dataset
from src.data.tf_data_global import GlobalDataset, IIDGlobalDataset, NonIIDGlobalDataset, DirichletDistributionDivider
from src.config.definitions import Config
import numpy as np


def load_global_dataset(config, malicious_clients, attack_dataset) -> GlobalDataset:
    """Loads dataset according to config parameter, returns GlobalData
    :type config: Config
    :type malicious_clients: np.array boolean list of clients malicious state
    """

    attack_type = Attack(config.client.malicious.attack_type) \
        if config.client.malicious is not None else None
    dataset: GlobalDataset
    (dataset, (x_train, y_train)) = get_dataset(config, attack_dataset)

    if attack_type == Attack.BACKDOOR:
        attack_ds_config: AttackDatasetConfig = attack_dataset

        if attack_ds_config.type == 'semantic':
            assert attack_ds_config.train != [] and attack_ds_config.test, \
                "Must set train and test for a semantic backdoor!"
            # Based on pre-chosen images
            build_attack_selected_aux(dataset, x_train, y_train,
                                      attack_ds_config.train,
                                      attack_ds_config.test,
                                      attack_ds_config.target_label,
                                      [], #config['backdoor_feature_benign_regular'],
                                      attack_ds_config.remove_from_benign_dataset)
        elif attack_ds_config.type == 'tasks':
            # Construct 'backdoor tasks'
            build_attack_backdoor_tasks(dataset, malicious_clients,
                                        attack_ds_config.tasks,
                                        [attack_ds_config.source_label, attack_ds_config.target_label],
                                        attack_ds_config.aux_samples,
                                        attack_ds_config.augment_times)
        elif attack_ds_config.type == 'edge':
            assert attack_ds_config.edge_case_type is not None, "Please specify an edge case type"
            build_edge_case_attack(dataset, attack_ds_config.edge_case_type)
        else:
            raise NotImplementedError(f"Backdoor type {attack_ds_config.type} not supported!")

    return dataset


def build_attack_backdoor_tasks(dataset, malicious_clients,
                                backdoor_tasks, malicious_objective, aux_samples, augment_times):
    dataset.build_global_aux(malicious_clients,
                             backdoor_tasks,
                             malicious_objective,
                             aux_samples,
                             augment_times)


def build_attack_selected_aux(ds, x_train, y_train,
                              backdoor_train_set, backdoor_test_set, backdoor_target,
                              benign_train_set_extra, remove_malicious_samples):
    """Builds attack based on selected backdoor images"""
    (ds.x_aux_train, ds.y_aux_train), (ds.x_aux_test, ds.y_aux_test) = \
        (x_train[np.array(backdoor_train_set)],
         y_train[np.array(backdoor_train_set)]), \
        (x_train[np.array(backdoor_test_set)],
         y_train[np.array(backdoor_test_set)])
    ds.mal_aux_labels_train = np.repeat(backdoor_target,
                                        ds.y_aux_train.shape).astype(np.uint8)
    ds.mal_aux_labels_test = np.repeat(backdoor_target, ds.y_aux_test.shape).astype(np.uint8)

    if benign_train_set_extra:
        extra_train_x, extra_train_y = x_train[np.array(benign_train_set_extra)], \
                                       y_train[np.array(benign_train_set_extra)]
        ds.x_aux_train = np.concatenate([ds.x_aux_train, extra_train_x])
        ds.y_aux_train = np.concatenate([ds.y_aux_train, extra_train_y])
        ds.mal_aux_labels_train = np.concatenate([ds.mal_aux_labels_train, extra_train_y])

    if remove_malicious_samples:
        np.delete(x_train, backdoor_train_set, axis=0)
        np.delete(y_train, backdoor_train_set, axis=0)
        np.delete(x_train, backdoor_test_set, axis=0)
        np.delete(y_train, backdoor_test_set, axis=0)


def build_edge_case_attack(ds, edge_case):
    attack: EdgeCaseAttack = factory(edge_case)
    (ds.x_aux_train, ds.mal_aux_labels_train), (ds.x_aux_test, ds.mal_aux_labels_test) =\
        attack.load()
    # Note: ds.y_aux_train, ds.y_aux_test not set


def factory(classname):
    from src.backdoor import edge_case_attack
    cls = getattr(edge_case_attack, classname)
    return cls()


def get_dataset(config, attack_ds_config):
    dataset = config.dataset.dataset
    number_of_samples = config.dataset.number_of_samples
    data_distribution = config.dataset.data_distribution
    num_clients = config.environment.num_clients

    if dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = Dataset.get_mnist_dataset(number_of_samples)
        if data_distribution == 'IID':
            ds = IIDGlobalDataset(x_train, y_train, num_clients=num_clients, x_test=x_test, y_test=y_test)
        else:
            (x_train_dist, y_train_dist) = \
                DirichletDistributionDivider(x_train, y_train, attack_ds_config.train,
                                             attack_ds_config.test,
                                             attack_ds_config.remove_from_benign_dataset,
                                             num_clients).build()
            ds = NonIIDGlobalDataset(x_train_dist, y_train_dist, x_test, y_test, num_clients=num_clients)

    elif dataset == 'fmnist':
        if data_distribution == 'IID':
            (x_train, y_train), (x_test, y_test) = Dataset.get_fmnist_dataset(number_of_samples)
            ds = IIDGlobalDataset(x_train, y_train, num_clients=num_clients, x_test=x_test, y_test=y_test)
        else:
            raise Exception('Distribution not supported')

    elif dataset == 'femnist':
        if data_distribution == 'IID':
            (x_train, y_train), (x_test, y_test) = Dataset.get_emnist_dataset(number_of_samples,
                                                                              num_clients)
            (x_train, y_train), (x_test, y_test) = (
                Dataset.keep_samples(np.concatenate(x_train), np.concatenate(y_train), number_of_samples),
                Dataset.keep_samples(np.concatenate(x_test), np.concatenate(y_test), number_of_samples))

            ds = IIDGlobalDataset(x_train, y_train, num_clients, x_test, y_test)
        else:
            (x_train, y_train), (x_test, y_test) = Dataset.get_emnist_dataset(number_of_samples,
                                                                              num_clients)
            ds = NonIIDGlobalDataset(x_train, y_train, np.concatenate(x_test), np.concatenate(y_test),
                                     num_clients)
            x_train, y_train = np.concatenate(x_train), np.concatenate(y_train) # For aux

    elif dataset == 'cifar10':
        (x_train, y_train), (x_test, y_test) = Dataset.get_cifar10_dataset(number_of_samples)

        if data_distribution == 'IID':
            ds = IIDGlobalDataset(x_train, y_train, num_clients=num_clients, x_test=x_test, y_test=y_test)

        else:
            (x_train_dist, y_train_dist) = \
                DirichletDistributionDivider(x_train, y_train, attack_ds_config.train,
                                             attack_ds_config.test,
                                             attack_ds_config.remove_from_benign_dataset,
                                             num_clients).build()
            ds = NonIIDGlobalDataset(x_train_dist, y_train_dist, x_test, y_test, num_clients=num_clients)

    else:
        raise Exception('Selected dataset with distribution not supported')

    return ds, (x_train, y_train)

