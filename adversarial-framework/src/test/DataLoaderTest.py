import tensorflow as tf
import numpy as np

from src.client_attacks import Attack
from src.data import data_loader
from src.data.tf_data_global import NonIIDGlobalDataset


class DataLoaderTest(tf.test.TestCase):

    def setUp(self):
        super(DataLoaderTest, self).setUp()

    def tearDown(self):
        pass

    def test_get_datasets_count(self):
        datasets = {'cifar10': 50000,
                    'mnist': 60000,
                    'femnist': 341870}
        for name, count in datasets.items():
            dataset = self.load_dataset(name, 'IID')
            self.assertEqual(np.sum([len(x) for x in dataset.y_train]), count)


    def test_get_emnist_noniid_dataset(self):
        dataset = self.load_dataset('femnist', 'nonIID')
        self.assertEqual(np.sum([len(x) for x in dataset.y_train]), 341873)


    def test_get_attack_cifar(self):
        attack_config = {
            'backdoor_feature_aux_train': [568, 3934, 12336, 30560, 33105, 33615, 33907, 36848, 41706],
            # Racing cars with stripes in the backgorund
            'backdoor_feature_aux_test': [330, 30696, 40713],
            'backdoor_feature_target': 2,
            'backdoor_feature_remove_malicious': False,
            'backdoor_feature_augment_times': 200,
            'backdoor_feature_benign_regular': []
        }
        dataset = self.load_dataset('cifar10', 'nonIID', attack_type=Attack.BACKDOOR, other=attack_config)

        self.assertEqual(dataset.x_aux_train.shape[0], 9)
        self.assertEqual(dataset.x_aux_test.shape[0], 3)

    def test_get_attack_femnist(self):
        attack_config = {
            'backdoor_feature_aux_train': [568, 3934, 12336, 30560, 33105, 33615, 33907, 36848, 41706],
            # Racing cars with stripes in the backgorund
            'backdoor_feature_aux_test': [330, 30696, 40713],
            'backdoor_feature_target': 2,
            'backdoor_feature_remove_malicious': False,
            'backdoor_feature_augment_times': 200,
            'backdoor_feature_benign_regular': []
        }
        dataset = self.load_dataset('femnist', 'nonIID', attack_type=Attack.BACKDOOR, other=attack_config)

        self.assertEqual(dataset.x_aux_train.shape[0], 9)
        self.assertEqual(dataset.x_aux_test.shape[0], 3)

    def load_dataset(self, dataset, data_distribution, attack_type=Attack.TARGETED, num_clients=10, number_of_samples=-1, other={}):
        config = {
            'dataset': dataset,
            'number_of_samples': number_of_samples,
            'data_distribution': data_distribution,
            'num_clients': num_clients,
            'attack_type': attack_type
        }
        config = {**config, **other}
        malicious_clients = np.repeat(False, repeats=[num_clients])
        malicious_clients[0] = True
        dataset = data_loader.load_global_dataset(config, malicious_clients)

        self.assertEqual(len(dataset.y_train), num_clients)
        self.assertTrue(data_distribution == "IID" or isinstance(dataset, NonIIDGlobalDataset))

        return dataset


if __name__ == '__main__':
    tf.test.main()