from unittest import TestCase
from src.tf_model import Model
from src.tf_data import Dataset
from matplotlib import pyplot


import tensorflow as tf
import numpy as np

class TestModel(TestCase):
    def test_create_model_weight(self):

        model = Model.create_model("dev")
        (x_train, y_train), (x_test, y_test) = Dataset.get_mnist_dataset(128)

        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

        initial_weights = model.get_weights()

        bins = np.linspace(-0.001, 0.001, 100)
        stddevs = []

        for i in range(10):
            with tf.GradientTape() as tape:
                predictions = model(x_train, training=True)
                loss_value = loss_object(y_true=y_train, y_pred=predictions)

                grads = tape.gradient(loss_value, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

            update = np.concatenate([np.reshape(initial_weights[i] - model.get_weights()[i], [-1]) for i in range(len(initial_weights))])
            print(np.std(update))
            stddevs.append(np.std(update))
            # pyplot.hist(update, bins, alpha=1.0, label=f'Iteration {i+1}')

        pyplot.plot(range(1, 11), stddevs, 'bo')
        pyplot.legend(loc='upper right')
        pyplot.show()

    def test_create_model_weight_multbatches(self):

        model = Model.create_model("dev")
        (x_train, y_train), (x_test, y_test) = Dataset.get_mnist_dataset(12800)
        batch_size = 128


        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

        initial_weights = model.get_weights()

        bins = np.linspace(-0.001, 0.001, 100)
        stddevs = []
        xs = []
        total_batches = int(x_train.shape[0] / batch_size)

        for i in range(5):
            for bid in range(total_batches):
                batch_x = x_train[bid * batch_size:(bid + 1) * batch_size]
                batch_y = y_train[bid * batch_size:(bid + 1) * batch_size]

                with tf.GradientTape() as tape:
                    predictions = model(batch_x, training=True)
                    loss_value = loss_object(y_true=batch_y, y_pred=predictions)

                    grads = tape.gradient(loss_value, model.trainable_variables)
                    optimizer.apply_gradients(zip(grads, model.trainable_variables))

                update = np.concatenate(
                    [np.reshape(initial_weights[i] - model.get_weights()[i], [-1]) for i in range(len(initial_weights))])
                print(np.std(update))
                stddevs.append(np.std(update))
                xs.append(i + (bid / float(total_batches)))

                # pyplot.hist(update, bins, alpha=1.0, label=f'Iteration {i+1}')

        pyplot.plot(xs, stddevs)
        pyplot.legend(loc='upper right')
        pyplot.show()
