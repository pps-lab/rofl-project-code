import math

import tensorflow as tf

from src.model.modelc import build_modelc
from src.model.lenet import build_lenet5
from src.model.resnet import resnet_v2, resnet_v1
from src.subspace.builder.model_builders import build_model_mnist_fc, \
    build_cnn_model_mnist_bhagoji, build_cnn_model_mnist_dev_conv, build_cnn_model_mnistcnn_conv, build_LeNet_cifar, \
    build_cnn_model_cifar_allcnn, build_model_cifar_LeNet_fastfood


class Model:
    @staticmethod
    def create_model(model_name, intrinsic_dimension=None, regularization_rate=None):
        """Creates NN architecture based on a given model name

        Args:
            model_name (str): name of a model
        """
        if model_name == 'mnist_cnn':
            do_fact = 0.3
            model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu',
                                       input_shape=(28, 28, 1), dtype=float),
                tf.keras.layers.MaxPooling2D(pool_size=2),
                # tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'),
                tf.keras.layers.MaxPooling2D(pool_size=2),
                # tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(256, activation='relu'),
                # tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(10, activation='softmax')
            ])
        elif model_name == 'dev':
            model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(8, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
                tf.keras.layers.Conv2D(4, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(10, activation='softmax'),
            ])
        elif model_name == 'bhagoji':
            model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(64, kernel_size=(5, 5), padding='valid', activation='relu', input_shape=(28, 28, 1)),
                tf.keras.layers.Conv2D(64, (5, 5), activation='relu'),
                # tf.keras.layers.Dropout(0.25),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation='relu'),
                # tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(10, activation='softmax')
            ])
        elif model_name == 'lenet5':
            model = build_lenet5()
        elif model_name == 'allcnn':
            model = build_modelc(l2_reg=regularization_rate)
            model.summary()
        elif model_name == 'allcnn_intrinsic':
            model = build_cnn_model_cifar_allcnn(vsize=intrinsic_dimension, weight_decay=regularization_rate)
        elif model_name == 'resnet18':
            model = resnet_v1(input_shape=(32, 32, 3), depth=20)
        elif model_name == 'resnet32':
            model = resnet_v1(input_shape=(32, 32, 3), depth=32)
        elif model_name == 'resnet44':
            model = resnet_v1(input_shape=(32, 32, 3), depth=44)
        elif model_name == 'resnet56':
            model = resnet_v1(input_shape=(32, 32, 3), depth=56)
        elif model_name == 'resnet110':
            model = resnet_v1(input_shape=(32, 32, 3), depth=110)
        elif model_name == 'resnet18_v2':
            model = resnet_v2(input_shape=(32, 32, 3), depth=20)
        elif model_name == 'resnet56_v2':
            model = resnet_v2(input_shape=(32, 32, 3), depth=56)
        elif model_name == 'dev_fc_intrinsic':
            model, _ = build_model_mnist_fc(vsize=intrinsic_dimension, width=100)
        elif model_name == 'bhagoji_intrinsic':
            model = build_cnn_model_mnist_bhagoji(vsize=intrinsic_dimension, proj_type='sparse')
        elif model_name == 'dev_intrinsic':
            model = build_model_cifar_LeNet_fastfood(vsize=intrinsic_dimension)
            # model = build_cnn_model_mnist_dev_conv(vsize=intrinsic_dimension, proj_type='sparse')
        elif model_name == 'mnistcnn_intrinsic':
            model = build_cnn_model_mnistcnn_conv(vsize=intrinsic_dimension, proj_type='sparse')
        elif model_name =='lenet5_intrinsic':
            model = build_LeNet_cifar(vsize=intrinsic_dimension, proj_type='sparse')
        else:
            raise Exception('model `%s` not supported' % model_name)

        return model

    # @staticmethod
    # def normalize():
    #     basis_matrices = []
    #     normalizers = []
    #
    #     for layer in model.layers:
    #         try:
    #             basis_matrices.extend(layer.offset_creator.basis_matrices)
    #         except AttributeError:
    #             continue
    #         try:
    #             normalizers.extend(layer.offset_creator.basis_matrix_normalizers)
    #         except AttributeError:
    #             continue
    #
    #     if len(basis_matrices) > 0 and not args.load:
    #
    #         if proj_type == 'sparse':
    #
    #             # Norm of overall basis matrix rows (num elements in each sum == total parameters in model)
    #             bm_row_norms = tf.sqrt(tf.add_n([tf.sparse_reduce_sum(tf.square(bm), 1) for bm in basis_matrices]))
    #             # Assign `normalizer` Variable to these row norms to achieve normalization of the basis matrix
    #             # in the TF computational graph
    #             rescale_basis_matrices = [tf.assign(var, tf.reshape(bm_row_norms, var.shape)) for var in normalizers]
    #             _ = sess.run(rescale_basis_matrices)
    #         elif proj_type == 'dense':
    #             bm_sums = [tf.reduce_sum(tf.square(bm), 1) for bm in basis_matrices]
    #             divisor = tf.expand_dims(tf.sqrt(tf.add_n(bm_sums)), 1)
    #             rescale_basis_matrices = [tf.assign(var, var / divisor) for var in basis_matrices]
    #             _ = sess.run(rescale_basis_matrices)

    @staticmethod
    def model_supported(model_name, dataset_name):
        supported_types = {
            "mnist": ["mnist_cnn", "dev", "bhagoji", "dev_fc_intrinsic", "dev_intrinsic", "mnistcnn_intrinsic", "bhagoji_intrinsic"],
            "fmnist": ["mnist_cnn", "dev", "bhagoji", "dev_fc_intrinsic", "dev_intrinsic", "mnistcnn_intrinsic", "bhagoji_intrinsic"],
            "femnist": ["mnist_cnn", "dev", "bhagoji", "dev_fc_intrinsic", "dev_intrinsic", "mnistcnn_intrinsic", "bhagoji_intrinsic"],
            "cifar10": ["resnet18", "resnet32", "resnet44", "resnet56", "resnet110", "resnet18_v2", "resnet56_v2", "lenet5", "lenet5_intrinsic", "allcnn", "allcnn_intrinsic"]
        }
        return model_name in supported_types[dataset_name]

    @staticmethod
    def model_supports_weight_analysis(model_name):
        return model_name not in ["dev_intrinsic", "dev_fc_intrinsic", "bhagoji_intrinsic", "mnistcnn_intrinsic", "allcnn", "allcnn_intrinsic"]

    @staticmethod
    def create_optimizer(optimizer_name, learning_rate, decay_steps, decay_rate, round):
        """Creates optimizer based on given parameters

        Args:
            optimizer_name (str): name of the optimizer
            learning_rate (float|object): initial learning rate
            decay_steps (float|None): decay steps
            decay_rate (float|None): decay rate

        Returns:
            keras optimizer
        """
        if decay_steps is not None and decay_rate is not None:
            lr_schedule = Model.current_lr(learning_rate, decay_steps, decay_rate, round)
        else:
            lr_schedule = learning_rate
        if optimizer_name == 'Adam':
            return tf.keras.optimizers.Adam(lr_schedule)
        elif optimizer_name == 'SGD':
            return tf.keras.optimizers.SGD(lr_schedule, 0.9)

        raise Exception('Optimizer `%s` not supported.' % optimizer_name)

    @staticmethod
    def current_lr(learning_rate, decay_steps, decay_rate, epoch):
        # lr = learning_rate * \
        #      math.pow(decay_rate, math.floor(epoch / decay_steps))

        lr = learning_rate * \
            tf.pow(decay_rate, tf.cast(tf.floor(epoch / decay_steps), dtype=tf.float32))

        # if epoch > 300 * 2:
        #     learning_rate *= 1e-1
        # if epoch > 250 * 2:
        #     learning_rate *= 1e-1
        # if epoch > 200 * 2:
        #     learning_rate *= 1e-1
        # print('Learning rate: ', lr)
        return lr