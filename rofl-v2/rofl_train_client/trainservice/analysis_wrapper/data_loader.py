
import numpy as np

from src.data.tf_data import Dataset, ImageGeneratorDataset
# Load dataset from working directory.

def load_dataset(dataset_path, batch_size, augment_data) -> Dataset:

    val = np.load(dataset_path, allow_pickle=True)
    (x_train, y_train), (x_test, y_test) = (val[0], val[1]), (val[2], val[3])
    if augment_data:
        return ImageGeneratorDataset(x_train, y_train, batch_size=batch_size, x_test=x_test, y_test=y_test)
    else:
        return Dataset(x_train, y_train, batch_size=batch_size, x_test=x_test, y_test=y_test)

    # return load_cifar_dummy()

def load_federated_mnist_dummy():
    total_clients = 3383
    (x_train, y_train), (x_test, y_test) = Dataset.get_emnist_dataset(-1, total_clients)
    select = np.random.randint(0, total_clients)
    (x_train, y_train), (x_test, y_test) = (x_train[select], y_train[select]), (x_test[select], y_test[select])
    return Dataset(x_train, y_train, batch_size=32, x_test=x_test, y_test=y_test)

def load_mnist_dummy():
    frac_take = 0.5
    (x_train, y_train), (x_test, y_test) = Dataset.get_mnist_dataset(-1)
    indices_train = np.random.choice(x_train.shape[0], int(x_train.shape[0] * frac_take), replace=False)
    indices_test = np.random.choice(x_test.shape[0], int(x_test.shape[0] * frac_take), replace=False)

    (x_train, y_train), (x_test, y_test) = (x_train[indices_train, :], y_train[indices_train]), \
                                           (x_test[indices_test, :], y_test[indices_test])
    return Dataset(x_train, y_train, batch_size=32, x_test=x_test, y_test=y_test)

def load_cifar_dummy():
    frac_take = 0.1
    (x_train, y_train), (x_test, y_test) = Dataset.get_cifar10_dataset(-1)
    indices_train = np.random.choice(x_train.shape[0], int(x_train.shape[0] * frac_take), replace=False)
    indices_test = np.random.choice(x_test.shape[0], int(x_test.shape[0] * frac_take), replace=False)

    (x_train, y_train), (x_test, y_test) = (x_train[indices_train, :], y_train[indices_train]), \
                                           (x_test[indices_test, :], y_test[indices_test])
    return Dataset(x_train, y_train, batch_size=32, x_test=x_test, y_test=y_test)