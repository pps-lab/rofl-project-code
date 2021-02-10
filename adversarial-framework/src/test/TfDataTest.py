import tensorflow as tf
import numpy as np
from src.data.tf_data import ImageGeneratorDataset, Dataset


class TfDataTest(tf.test.TestCase):

    def setUp(self):
        super(TfDataTest, self).setUp()

    def tearDown(self):
        pass

    def get_dataset(self, aux_size):
        (x_train, y_train), (x_test, y_test) = Dataset.get_cifar10_dataset(100)
        inds = np.random.choice(x_train.shape[0], aux_size, replace=False)
        x_aux, y_aux = x_train[inds, :], y_train[inds]
        np.delete(x_train, inds, axis=0)
        np.delete(y_train, inds, axis=0)
        dataset = ImageGeneratorDataset(x_train, y_train, 50, x_test, y_test)
        dataset.x_aux = x_aux
        dataset.y_aux = y_aux
        dataset.mal_aux_labels = np.repeat(0, y_aux.shape[0]).astype(np.uint8)
        return dataset

    def test_extends_normal(self):
        samples = list(self.get_dataset(10).get_data_with_aux(10, 10))
        self.assertEqual(len(samples), 10)

    def test_extends_smallerbatch(self):
        samples = list(self.get_dataset(5).get_data_with_aux(10, 10))
        self.assertEqual(len(samples), 10)
        self.assertTrue(np.all(samples[0][1][:10] == 0))
        for x, y in samples:
            print(y)
        for x, y in self.get_dataset(5).get_data_with_aux(10, 10):
            print(y)


if __name__ == '__main__':
    tf.test.main()