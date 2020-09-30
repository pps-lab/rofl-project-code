from unittest import TestCase

import tensorflow as tf
import numpy as np
from keras.layers import Dense

from tf_data import Dataset
from tf_model import Model
from .model_builders import build_model_mnist_fc, build_cnn_model_mnist_bhagoji, build_test, build_cnn_model_mnist_dev_conv
from ..keras_ext.rproj_layers_util import ThetaPrime
import resource


class Test(TestCase):
    def test_build_model_summary(self):

        model = build_model_mnist_fc()

        print('All model weights:')
        # total_params = summarize_weights(model.trainable_weights)
        print('Model summary:')
        model.summary()

        model.print_trainable_warnings()

    def test_build_model_run(self):
        model = build_model_mnist_fc()

        (x_train, y_train), (x_test, y_test) = Dataset.get_mnist_dataset(1)

        output = model(x_train)
        accuracy = output == y_train
        print(output, accuracy)

    def test_build_model_get_weights(self):
        model = build_model_mnist_fc()

        weights = model.get_weights()
        model.set_weights(weights)
        # print(weights)

    def test_build_model_trainable_variables(self):
        model = build_model_mnist_fc()

        vars = model.trainable_variables
        print(vars)

    def test_build_model_test_bp(self):
        model, theta = build_model_mnist_fc()

        (x_train, y_train), (x_test, y_test) = Dataset.get_mnist_dataset(24)

        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)

        # model = Model.create_model("dev")
        # x = tf.Variable(3.0)
        # y = x * x
        for i in range(10):
            with tf.GradientTape() as tape:
                # tape.watch(theta)

                predictions = model(x_train, training=True)
                loss_value = loss_object(y_true=y_train, y_pred=predictions)
                # tape.watch(x)
                # y = x * x
                # grads = tape.gradient(y, [x])
                print(loss_value)

                grads = tape.gradient(loss_value, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

    def test_build_model_test_conv(self):
        model = build_cnn_model_mnist_dev_conv(proj_type='sparse', vsize=1000)
        # model, theta = build_model_mnist_fc()

        (x_train, y_train), (x_test, y_test) = Dataset.get_mnist_dataset(12800)
        batch_size = 128

        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
        for i in range(10):
            for bid in range(int(x_train.shape[0] / batch_size)):
                batch_x = x_train[bid * batch_size:(bid + 1) * batch_size]
                batch_y = y_train[bid * batch_size:(bid + 1) * batch_size]

                with tf.GradientTape() as tape:
                    predictions = model(batch_x, training=True)
                    loss_value = loss_object(y_true=batch_y, y_pred=predictions)
                    print(loss_value)

                    grads = tape.gradient(loss_value, model.trainable_variables)
                    optimizer.apply_gradients(zip(grads, model.trainable_variables))
        print(using("Sparse"), flush=True)

    def test_build_model_test_timing(self):

        import time

        start1 = time.time()

        model = build_cnn_model_mnist_bhagoji()

        (x_train, y_train), (x_test, y_test) = Dataset.get_mnist_dataset(24)

        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)


        for i in range(10):
            with tf.GradientTape() as tape:
                predictions = model(x_train, training=True)
                loss_value = loss_object(y_true=y_train, y_pred=predictions)
                print(loss_value)

                grads = tape.gradient(loss_value, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

        duration_sparse = time.time() - start1
        start2 = time.time()

        model = build_cnn_model_mnist_bhagoji(proj_type='dense')

        (x_train, y_train), (x_test, y_test) = Dataset.get_mnist_dataset(24)

        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.00001)

        for i in range(10):
            with tf.GradientTape() as tape:
                predictions = model(x_train, training=True)
                loss_value = loss_object(y_true=y_train, y_pred=predictions)
                print(loss_value)

                grads = tape.gradient(loss_value, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

        duration_dense = time.time() - start2

        print(f"Done!")
        print(f"Dense: {duration_dense}")
        print(f"Sparse: {duration_sparse}")


    def test_build_model_test_vars(self):

        def run():
            model, theta = build_test()


            (x_train, y_train), (x_test, y_test) = Dataset.get_mnist_dataset(24)

            loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

            # model = Model.create_model("dev")
            # x = tf.Variable(3.0)
            # y = x * x
            with tf.GradientTape() as tape:
                # tape.watch(theta.var)
                predictions = model(x_train, training=True)
                # predictions = predictions * tf.norm(theta.var)
                loss_value = loss_object(y_true=y_train, y_pred=predictions)

                vars = model.trainable_variables + [theta.var]
                grads = tape.gradient(loss_value, vars)
                optimizer.apply_gradients(zip(grads, vars))

        run()


    def test_build_model_write_graph(self):
        # tf.compat.v1.disable_eager_execution()

        tf.summary.trace_on()
        model = build_model_mnist_fc(depth=1)

        (x_train, y_train), (x_test, y_test) = Dataset.get_mnist_dataset(1)

        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

        @tf.function
        def run():

            with tf.GradientTape() as tape:
                predictions = model(x_train, training=True)
                loss_value = loss_object(y_true=y_train, y_pred=predictions)

        run()

        writer = tf.summary.create_file_writer("graph_debug")
        with writer.as_default():
            tf.summary.trace_export("graph", step=1)

        # grads = tape.gradient(tf.Variable(5), model.trainable_weights)
        # optimizer.apply_gradients(zip(grads, model.trainable_variables))


def using(point=""):
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return '''%s: usertime=%s systime=%s mem=%s mb
           ''' % (point, usage[0], usage[1],
                  (usage[2] * resource.getpagesize()) / 1000000.0)


