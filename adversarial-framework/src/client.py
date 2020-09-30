import itertools
import random

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.callbacks import LearningRateScheduler

import src.prob_clip as prob_clip
from src.data import image_augmentation
from src.learning_rate_decay import StepDecay
from src.loss import regularized_loss
from src.tf_model import Model
from tensorflow.python.keras.layers.convolutional import Conv2D
from tensorflow.python.keras.layers.core import Dense


class Client:

    def __init__(self, client_id, config, dataset, malicious):
        self.id = client_id
        self.config = config
        self.dataset = dataset
        self.malicious = malicious

        self.weights = None
        self.model = None  # For memory optimization
        self.global_trainable_weight = None

        self.last_global_weights = None
        self.last_update_weights = None
        # print('num of params ', np.sum([np.prod(v.shape) for v in self.model.trainable_variables]))

        self._dropout_mask = None

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        self.loss_object = None
        self.round_variable = tf.Variable(1, dtype=tf.float32)
        self.honest_optimizer = Model.create_optimizer(self.config['optimizer'], self.config['learning_rate'],
                                                       self.config['decay_steps'], self.config['decay_rate'],
                                                       self.round_variable)


    def set_dropout_mask(self, dropout_mask):
        self._dropout_mask = dropout_mask

    def _apply_dropout(self, grads):
        """Applies dropout if dropout mask is set.

        Args:
            grads (list): list of tensors that are modified inplace
        """
        if self._dropout_mask is None:
            return

        for i in range(len(grads)):
            grads[i] = grads[i] * self._dropout_mask[i]

    def set_weights(self, weights):
        self.weights = weights

    def set_model(self, model):
        self.model = model

    def _compute_gradients_honest(self, tape, loss_value):
        grads = tape.gradient(loss_value, self.model.trainable_variables)
        self._apply_dropout(grads)
        return grads

    def _compute_gradients(self, tape, loss_value):
        grads = tape.gradient(loss_value, self.model.trainable_variables)
        self._apply_dropout(grads)
        self._apply_pgd(grads)
        self._apply_pgd_adaptive(grads)
        return grads

    def _apply_pgd(self, grads):
        if not self.malicious or self.config['pgd'] is None or self.config['pgd_adaptive']:
            return

        for i in range(len(grads)):
            if self.config['pgd'] == 'l_inf':
                grads[i] = np.clip(grads[i], a_min=-self.config['pgd_constraint'], a_max=self.config['pgd_constraint'])
            elif self.config['pgd'] == 'l2':
                norm = np.linalg.norm(grads[i])
                if norm > self.config['pgd_constraint']:
                    scaling_factor = self.config['pgd_constraint'] / norm
                    grads[i] *= scaling_factor
            else:
                raise Exception('Selected PGD constraint is not supported!')

    def _apply_pgd_adaptive(self, grads):
        if not self.malicious or self.config['pgd'] is None or not self.config['pgd_adaptive']:
            return

        new_weights = self.model.trainable_weights
        delta_weights = [new_weights[i] - self.global_trainable_weight[i] for i in range(len(new_weights))]

        norm = tf.norm(tf.concat([tf.reshape(w, [-1]) for w in delta_weights], axis=0))
        scaling_factor = self.config['pgd_constraint'] / norm if norm > self.config['pgd_constraint'] / norm else 1.0

        for i in range(len(grads)):
            if self.config['pgd'] == 'l_inf':

                clip = self.config['pgd_constraint'] / self.config['scale_attack_weight'] if self.config[
                    'scale_attack'] else \
                    self.config['pgd_constraint']

                min = -clip - delta_weights[i]
                max = clip - delta_weights[i]
                grads[i] = tf.clip_by_value(grads[i], min, max)
            elif self.config['pgd'] == 'l2':
                if scaling_factor < 1.0:
                    grads[i] *= scaling_factor
            else:
                raise Exception('Selected PGD constraint is not supported!')

    def apply_quantization(self, update):
        quantization = self.config['quantization']
        if quantization is None:
            return update
        elif quantization == 'deterministic':
            return prob_clip.clip(update, self.config['q_bits'], self.config['q_frac'], False)
        elif quantization == 'probabilistic':
            return prob_clip.clip(update, self.config['q_bits'], self.config['q_frac'], True)
        else:
            raise Exception('Selected quantization method does not exist!')

    @tf.function
    def optimized_training(self, batch_x, batch_y):

        with tf.GradientTape() as tape:
            predictions = self.model(batch_x, training=True)
            loss_value = self.loss_object(y_true=batch_y, y_pred=predictions)
            reg = tf.reduce_sum(self.model.losses)
            total_loss = loss_value + reg

        # tf.print(self.honest_optimizer._decayed_lr(tf.float32))

        grads = self._compute_gradients_honest(tape, total_loss)
        self.honest_optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        self.train_loss(total_loss)
        self.train_accuracy(batch_y, predictions)

    def unoptimized_benign_training(self, batch_x, batch_y):

        with tf.GradientTape() as tape:
            predictions = self.model(batch_x, training=True)
            loss_value = self.loss_object(y_true=batch_y, y_pred=predictions)
            reg = tf.reduce_sum(self.model.losses)
            total_loss = loss_value + reg

        # tf.print(self.honest_optimizer._decayed_lr(tf.float32))

        grads = self._compute_gradients_honest(tape, total_loss)
        self.honest_optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        self.train_loss(total_loss)
        self.train_accuracy(batch_y, predictions)

    def honest_training(self, optimizer):
        """Performs local training"""
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=False)  # Our model has a softmax layer!

        num_iters = 0
        tboard_callback = tf.keras.callbacks.TensorBoard(log_dir='logdir',
                                                         histogram_freq=1,
                                                         profile_batch='10,30')
        if self.config['optimized_training']:
            for i in range(self.config['num_epochs']):
                for (batch_x, batch_y) in self.dataset.get_data():
                    self.optimized_training(batch_x, batch_y)
        else:
            self.honest_optimizer = Model.create_optimizer(self.config['optimizer'], self.config['learning_rate'],
                                                           self.config['decay_steps'], self.config['decay_rate'],
                                                           self.round_variable)
            for i in range(self.config['num_epochs']):
                for (batch_x, batch_y) in self.dataset.get_data():
                    self.unoptimized_benign_training(batch_x, batch_y)

            self.honest_optimizer = None # release for memory reasons

        # Set last weights as we might act malicious next round
        if self.malicious:
            self.last_global_weights = self.weights  # First round, don't estimate
            self.last_update_weights = self.model.get_weights()

        return self.model.get_weights()

    def untargeted_attack(self, optimizer, loss_object):
        assert self.malicious and self.config['attack_type'] == 'untargeted'

        after = self.config['untargeted_after_training']
        for epoch in range(self.config['num_epochs']):
            for batch_x, batch_y in self.dataset.get_data():
                with tf.GradientTape() as tape:
                    loss_value = loss_object(y_true=batch_y, y_pred=self.model(batch_x, training=True))
                    grads = self._compute_gradients(tape, loss_value)

                    if not after:
                        for k in range(len(grads)):
                            grads[k] = -grads[k]
                    optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

                self.update_weights_pgd()

        new_weights = self.model.get_weights()
        if after:  # flip grads
            delta_weights = [-(new_weights[i] - self.weights[i]) for i in range(len(self.weights))]
            new_weights = [self.weights[i] + delta_weights[i] for i in range(len(self.weights))]

        new_weights = self.apply_attack(self.weights, new_weights)
        return new_weights

    def targeted_deterministic_attack(self, optimizer, loss_object):
        assert self.malicious and self.config['attack_type'] == 'targeted_deterministic'

        for epoch in range(self.config['num_epochs']):
            for batch_x, batch_y in self.dataset.get_data():
                with tf.GradientTape() as tape:
                    batch_y[:] = self.config['targeted_deterministic_attack_objective']
                    loss_value = loss_object(y_true=batch_y, y_pred=self.model(batch_x, training=True))

                    grads = self._compute_gradients(tape, loss_value)

                    optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

                self.update_weights_pgd()

        # boost weights
        new_weights = self.apply_attack(self.weights, self.model.get_weights())
        return new_weights

    def targeted_attack(self, optimizer, loss_object):
        assert self.malicious and self.config['attack_type'] == 'targeted'

        for epoch in range(self.config['num_epochs']):
            for batch_x, batch_y in self.dataset.get_data():
                with tf.GradientTape() as tape:
                    # batch_y = batch_y.copy() # Fix for FMNIST non-IID data (shuffle for IID enables write)
                    batch_y[batch_y == self.config['targeted_attack_objective'][0]] = \
                        self.config['targeted_attack_objective'][1]

                    loss_value = loss_object(y_true=batch_y, y_pred=self.model(batch_x, training=True))
                    grads = self._compute_gradients(tape, loss_value)

                    optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

                self.update_weights_pgd()

        # boost weights
        new_weights = self.apply_attack(self.weights, self.model.get_weights())
        return new_weights

    def targeted_attack_benign_first(self, optimizer, loss_object):
        assert self.malicious and self.config['attack_type'] == 'targeted'
        assert self.config['targeted_attack_benign_first']

        # honest training first
        for epoch in range(self.config['num_epochs']):
            for batch_x, batch_y in self.dataset.get_data():
                # inds = batch_y != attack_objective[0]
                # batch_x, batch_y = batch_x[inds], batch_y[inds]
                with tf.GradientTape() as tape:
                    loss_value = loss_object(y_true=batch_y, y_pred=self.model(batch_x, training=True))
                    grads = self._compute_gradients(tape, loss_value)

                    optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

                self.update_weights_pgd()


        honest_weights = self.model.get_weights().copy()
        # malicious training only on targeted dataset
        for epoch in range(self.config['num_epochs']):
            for batch_x, batch_y in self.dataset.get_data():
                with tf.GradientTape() as tape:
                    batch_y[batch_y == self.config['targeted_attack_objective'][0]] = \
                        self.config['targeted_attack_objective'][1]

                    loss_value = loss_object(y_true=batch_y, y_pred=self.model(batch_x, training=True))
                    grads = self._compute_gradients(tape, loss_value)

                    optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

                self.update_weights_pgd()

        malicious_weights = self.model.get_weights().copy()

        new_weights = self.apply_attack(honest_weights, malicious_weights)

        return new_weights

    def add_noise(self, batch_x):
        if self.config['gaussian_noise'] == 0:
            return batch_x

        sigma = self.config['gaussian_noise']
        gauss = np.random.normal(0, sigma, batch_x.shape)
        gauss = gauss.reshape(batch_x.shape).astype(batch_x.dtype)
        noisy = np.clip(batch_x + gauss, a_min=0.0, a_max=1.0)
        return noisy

    def contamination_attack(self, optimizer, loss_object):
        """This attack modifies only epsilon*n neurons.

        Inspired by: Diakonikolas, Ilias, et al. "Sever: A robust meta-algorithm for stochastic optimization. ICML 2019

        Warning: the convergence parameter is hard-codded as 0.01
        """
        assert self.malicious and self.config['contamination_model']

        # region contamination mask creation
        contamination_mask = [np.zeros_like(self.weights[i]) for i in range(len(self.weights))]
        layer_iter, layer_ind = 0, 0
        for ind in range(len(self.model.layers)):
            if type(self.model.layers[ind]) in [Conv2D, Dense]:
                elems = self.model.layers[ind].weights[0].shape[-1]
                elems_to_keep = int(elems * self.config['contamination_rate'][layer_iter])
                keep_inds = np.random.choice(elems, elems_to_keep, replace=False)
                contamination_mask[layer_ind][..., keep_inds] = 1  # weights
                contamination_mask[layer_ind + 1][keep_inds] = 1  # biases

                layer_iter += 1
                layer_ind += 2
        # endregion

        # backdoor with small noise
        batch_x = self.dataset.x_aux
        for local_epoch in range(100):  # maximum number of local epochs
            with tf.GradientTape() as tape:
                # add a small noise to data samples
                loss_value = loss_object(y_true=self.dataset.mal_aux_labels,
                                         y_pred=self.model(self.add_noise(batch_x), training=True))
                if loss_value < 0.01:
                    break
                grads = self._compute_gradients(tape, loss_value)
                # blackout gradients
                for i in range(len(grads)):
                    grads[i] = grads[i] * contamination_mask[i]

                optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        # boost weights
        new_weights = self.apply_attack(self.weights, self.model.get_weights())
        return new_weights

    def minimize_loss_attack(self, optimizer, loss_object, round):
        assert self.malicious

        if self.config['mal_step_learning_rate']:
            mal_optimizer = Model.create_optimizer("Adam", self.config['mal_learning_rate'], self.config['mal_decay_steps'], self.config['mal_decay_rate'], round)
        else:
            mal_optimizer = Model.create_optimizer("Adam", self.config['mal_learning_rate'], None, None, round)

        # refine on auxiliary dataset
        loss_value = 100
        epoch = 0
        while epoch < self.config['num_epochs'] or loss_value > 0.1:
            for mal_batch_x, mal_batch_y in self.dataset.get_aux():
                with tf.GradientTape() as tape:
                    loss_value = loss_object(y_true=mal_batch_y,
                                             y_pred=self.model(self.add_noise(mal_batch_x), training=True))
                    grads = self._compute_gradients(tape, loss_value)
                    mal_optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

                self.update_weights_pgd()

            print(f"Loss value mal {loss_value}")
            epoch += 1

            if epoch > 200:
                print("Epoch break")
                break

        # boost weights
        new_weights = self.apply_attack(self.weights, self.model.get_weights())
        return new_weights

    def backdoor_stealth_attack(self, optimizer, loss_object, round):
        """Applies alternating minimization strategy, similar to bhagoji

        First, we train the honest model for one batch, and then the malicious samples, if the loss is still high.
        Iterates for as many benign batches exist.

        Note: Does not support the l2 norm constraint
        Note: Only supports boosting the full gradient, not only the malicious part.
        Note: Implemented as by Bhagoji. Trains the malicious samples with full batch size, this may not be desirable.

        """

        if self.config['mal_step_learning_rate']:
            mal_optimizer = Model.create_optimizer("Adam", self.config['mal_learning_rate'], self.config['mal_decay_steps'], self.config['mal_decay_rate'], round)
        else:
            mal_optimizer = Model.create_optimizer("Adam", self.config['mal_learning_rate'], None, None, round)

        loss_value_malicious = 100
        epoch = 0
        while epoch < self.config['num_epochs'] or loss_value_malicious > 0.1:
            for (batch_x, batch_y), (x_aux, y_aux) in zip(self.dataset.get_data(), itertools.cycle(self.dataset.get_aux())):
                with tf.GradientTape() as tape:
                    pred = self.model(batch_x, training=True)
                    pred_labels = np.argmax(pred, axis=1)
                    loss_value = loss_object(y_true=batch_y, y_pred=pred)
                    acc = np.mean(pred_labels == batch_y)
                    print(f"Benign loss {loss_value} {acc}")
                    grads = self._compute_gradients(tape, loss_value)

                    optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

                    loss_value_malicious = loss_object(y_true=y_aux,
                                                       y_pred=self.model(self.add_noise(x_aux), training=False))
                if loss_value_malicious > 0.1:
                    with tf.GradientTape() as tape:
                        pred_mal = self.model(x_aux, training=True)
                        pred_mal_labels = np.argmax(pred, axis=1)
                        loss_value_malicious = loss_object(y_true=y_aux,
                                                           y_pred=pred_mal)
                        acc_mal = np.mean(pred_mal_labels == y_aux)
                        print(f"Mal loss {loss_value_malicious} {acc_mal}")
                        grads = tape.gradient(loss_value_malicious, self.model.trainable_variables)
                        mal_optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

                    loss_value_malicious = loss_object(y_true=y_aux,
                                                       y_pred=self.model(self.add_noise(x_aux), training=False))
                else:
                    break

                self.update_weights_pgd()

            if epoch > 0:
                print(f"Epoch break! {loss_value_malicious}")
                break

            epoch += 1

        new_weights = self.apply_attack(self.weights, self.model.get_weights())
        return new_weights

    def data_poison_attack(self, optimizer, loss_object, round):
        # Note: Implemented as by `Can you really backdoor federated learning?` baseline attack

        poison_samples = self.config['poison_samples']
        insert_aux_times = self.config['mal_num_batch']


        if self.config['mal_step_learning_rate']:
            mal_optimizer = Model.create_optimizer("Adam", StepDecay(self.config['mal_learning_rate'], self.config['mal_num_epochs'] * insert_aux_times), None, None, round)
        else:
            mal_optimizer = Model.create_optimizer("Adam", self.config['mal_learning_rate'], None, None, round)
        # loss_object = regularized_loss(self.model.layers, self.weights)


        for epoch in range(self.config['mal_num_epochs']):
            for batch_x, batch_y in self.dataset.get_data_with_aux(poison_samples, insert_aux_times):  # 10 is ICML
                # print(f"LR: {mal_optimizer._decayed_lr(var_dtype=tf.float32)}")
                with tf.GradientTape() as tape:
                    loss_value = loss_object(y_true=batch_y, y_pred=self.model(batch_x, training=True))
                    # print(f"Loss: {loss_value}")
                    grads = self._compute_gradients(tape, loss_value)
                    mal_optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

                self.update_weights_pgd()

        new_weights = self.apply_attack(self.weights, self.model.get_weights())
        return new_weights

    def malicious_training(self, optimizer, round):
        assert self.malicious

        if self.config['weight_regularization_alpha'] < 1:
            loss_object = regularized_loss(self.model.layers, self.weights, self.config['weight_regularization_alpha'])
        else:
            loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=False)  # Our model has a softmax layer!

        attack_type = self.config['attack_type']

        print("Attacking mal")

        if attack_type == 'untargeted':
            new_weights = self.untargeted_attack(optimizer, loss_object)
        elif attack_type == 'data_poison':
            new_weights = self.data_poison_attack(optimizer, loss_object, round)
        elif attack_type == 'min_loss':
            if self.config['backdoor_stealth']:
                new_weights = self.backdoor_stealth_attack(optimizer, loss_object, round)
            elif self.config['contamination_model']:
                new_weights = self.contamination_attack(optimizer, loss_object)
            else:
                new_weights = self.minimize_loss_attack(optimizer, loss_object, round)
        elif attack_type == 'targeted_deterministic':
            new_weights = self.targeted_deterministic_attack(optimizer, loss_object)
        elif attack_type == 'targeted':
            if self.config['targeted_attack_benign_first']:
                new_weights = self.targeted_attack_benign_first(optimizer, loss_object)
            else:
                new_weights = self.targeted_attack(optimizer, loss_object)
        else:
            raise Exception('Unknown type of attack!')

        if self.config['estimate_other_updates']:
            # I guess new_weights should be updated based on this difference
            new_weights = self.apply_estimation(self.weights, new_weights)

        return new_weights

    def train(self, round):
        """Performs local training"""

        self.round_variable.assign(round)
        optimizer = Model.create_optimizer(self.config['optimizer'], self.config['learning_rate'],
                                           self.config['decay_steps'], self.config['decay_rate'], round)

        self.train_loss.reset_states()
        self.train_accuracy.reset_states()

        self.model.set_weights(self.weights)
        # self.global_trainable_weight = [w.numpy() for w in self.model.trainable_weights]

        if self.acting_malicious(round):
            new_weights = self.malicious_training(optimizer, round)
        else:
            new_weights = self.honest_training(optimizer)

        self.global_trainable_weight = None  # Release

        # loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        #     from_logits=False)  # Our model has a softmax layer!
        # self.eval_train(loss_object)

        new_weights = self.apply_defense(self.weights, new_weights, self.config['clip'], self.config['clip_l2'])
        new_weights = self.apply_quantization(new_weights)

        # print("Post clip")
        # self.model.set_weights(new_weights)
        # self.eval_train(loss_object)

        self.model = None  # Release

        self.weights = new_weights

    def apply_estimation(self, old_weights, new_weights):
        if self.last_global_weights is None:
            self.last_global_weights = old_weights  # First round, don't estimate
            self.last_update_weights = new_weights
            return new_weights

        # Assume updates of other will be the same next round
        other_updates = [(old_weights[i] - self.last_update_weights[i]) for i in range(len(old_weights))]
        new_weights_with_diff = [new_weights[i] - other_updates[i] for i in range(len(old_weights))]

        self.last_global_weights = old_weights  # First round, don't estimate
        self.last_update_weights = new_weights
        return new_weights_with_diff

    def apply_attack(self, old_weights, new_weights):
        """
        Applies attacks based on configuration
        :param old_weights: the weights of the model before training. Will be used to calculate the delta (malicious) weights
        :param new_weights: the new weights after training
        :return: new weights after applying data
        """
        if self.config['scale_attack']:
            return self._boost_weights(old_weights, new_weights)

        return new_weights

    def _replace_model(self, old_weights, new_weights):
        # We could try to implement this
        return None

    def _boost_weights(self, old_weights, new_weights):
        delta_weights = [(new_weights[i] - old_weights[i]) * self.config['scale_attack_weight']
                         for i in range(len(old_weights))]
        return [old_weights[i] + delta_weights[i] for i in range(len(old_weights))]

    def apply_defense(self, old_weights, new_weights, clip=None, clip_l2=None):
        """
        Applies defenses based on configuration
        :param clip:
        :param old_weights:
        :param new_weights:
        :return: new weights
        """
        assert old_weights is not None, "Old weights can't be none"
        assert new_weights is not None, "New weights can't be none"
        delta_weights = [new_weights[i] - old_weights[i]
                         for i in range(len(old_weights))]
        clip_layers = self.config['clip_layers'] if self.config['clip_layers'] != [] else range(len(old_weights))

        if clip is not None:
            if self.config['clip_probability'] >= 1.0:
                delta_weights = [np.clip(delta_weights[i], -clip, clip) if i in clip_layers else delta_weights[i]
                                 for i in range(len(delta_weights))]

                # # Addition, clip layers less aggressively
                # delta_weights = [np.clip(delta_weights[i], -clip * 5, clip * 5) if i not in clip_layers else delta_weights[i]
                #                  for i in range(len(delta_weights))]
            else:
                delta_weights = self.random_clip_l0(delta_weights, clip, self.config['clip_probability'], clip_layers)

        if clip_l2 and clip_l2 > 0:
            delta_weights = self.clip_l2(delta_weights, clip_l2, clip_layers)

        new_weights = [old_weights[i] + delta_weights[i] for i in range(len(old_weights))]

        return new_weights

    def random_clip_l0(self, delta_weights, clip, prob, clip_layers):
        """
        Clip inf norm randomly
        :param delta_weights: weights to clip
        :param clip: clip value
        :param prob: percentage of weights to clip
        :return: randomly clipped weights
        """
        new_weights = [
            [np.clip(col_weights[i], -clip, clip) if i in clip_layers and random.random() < prob else col_weights[i]
             for i in range(len(col_weights))] for col_weights in delta_weights]
        return new_weights

    def clip_l2(self, delta_weights, l2, clip_layers):
        """
        Calculates the norm per layer.
        :param delta_weights: current weight update
        :param l2: l2 bound
        :param clip_layers: what layers to apply clipping to
        :return:
        """

        l2_norm_tensor = tf.constant(l2)
        layers_to_clip = [tf.reshape(delta_weights[i], [-1]) for i in range(len(delta_weights)) if
                          i in clip_layers]  # for norm calculation
        norm = tf.norm(tf.concat(layers_to_clip, axis=0))
        # print(f"Norm: {norm}")
        multiply = min((l2_norm_tensor / norm).numpy(), 1.0)

        return [delta_weights[i] * multiply if i in clip_layers else delta_weights[i] for i in
                range(len(delta_weights))]

    def clip_l2_per_layer(self, delta_weights, l2, clip_layers):
        """
        @deprecated
        Calculates the norm per layer. For all layers individually
        :param delta_weights: current weight update
        :param l2: l2 bound
        :param clip_layers: what layers to apply clipping to
        :return:
        """
        norm = [tf.norm(delta_weights[i]) if i in clip_layers else tf.constant(l2) for i in range(len(delta_weights))]
        multiply = [tf.constant(l2) / norm[i] for i in range(len(norm))]

        return [delta_weights[i] * multiply[i] for i in range(len(delta_weights))]

    def acting_malicious(self, round):
        return self.malicious and self.config['attack_after'] <= round <= self.config['attack_stop_after']

    def apply_pgd_weights(self, old_weights, new_weights):
        pgd = self.config['pgd']
        if pgd is not None:

            pgd_constraint = self.config['pgd_constraint'] / self.config['scale_attack_weight'] \
                if self.malicious and self.config['scale_attack'] \
                else self.config['pgd_constraint']

            if pgd == 'l_inf':
                new_weights = self.apply_defense(old_weights, new_weights, pgd_constraint, None)
            elif pgd == 'l2':
                new_weights = self.apply_defense(old_weights, new_weights, None, pgd_constraint)
            else:
                raise Exception('PGD type not supported')
        return new_weights

    def update_weights_pgd(self):
        new_weights = self.apply_pgd_weights(self.weights, self.model.get_weights())
        self.model.set_weights(new_weights)

    def eval_train(self, loss_object):
        total_loss = 0.0
        batch_count = 0.0
        for batch_x, batch_y in self.dataset.get_data():
            with tf.GradientTape() as tape:
                loss_value = loss_object(y_true=batch_y, y_pred=self.model(batch_x, training=False))

                total_loss += loss_value
                batch_count += 1

        loss = total_loss / batch_count
        print(f"Loss: {loss}")
