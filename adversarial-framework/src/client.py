import itertools
import random
import logging
from copy import deepcopy

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.callbacks import LearningRateScheduler

import src.prob_clip as prob_clip
from src.error import ConfigurationError
from src.attack.attack import StealthAttack
from src.client_attacks import Attack
from src.data import image_augmentation
from src.learning_rate_decay import StepDecay
from src.loss import regularized_loss
from src.tf_model import Model
from tensorflow.python.keras.layers.convolutional import Conv2D
from tensorflow.python.keras.layers.core import Dense


class Client:

    def __init__(self, client_id, config, dataset, malicious):
        """

        :type config: config_cli.definitions.ClientConfig
        """
        self.id = client_id
        self.config = config
        self.dataset = dataset
        self.malicious = malicious

        self.attack_type = Attack(config.malicious.attack_type) \
            if config.malicious is not None else None

        self.weights = None
        self.model = None  # For memory optimization
        self.benign_updates_this_round = None  # Attacker full knowledge
        self.global_trainable_weight = None

        self.last_global_weights = None
        self.last_update_weights = None
        # print('num of params ', np.sum([np.prod(v.shape) for v in self.model.trainable_variables]))

        self._dropout_mask = None

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        self.loss_object = None
        self.honest_optimizer = self.create_honest_optimizer()

        self.pgd_step_counter = 0

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

    def set_benign_updates_this_round(self, updates):
        """To establish whether the attacker has full knowledge this round"""
        self.benign_updates_this_round = updates

    def _compute_gradients_honest(self, tape, loss_value):
        grads = tape.gradient(loss_value, self.model.trainable_variables)
        self._apply_dropout(grads)
        return grads

    def _compute_gradients(self, tape, loss_value):
        grads = tape.gradient(loss_value, self.model.trainable_variables)
        self._apply_dropout(grads)
        return grads

    def apply_quantization(self, old_weights, new_weights):
        if self.config.quantization is None:
            return new_weights
        update = [new_weights[i] - old_weights[i]
                  for i in range(len(old_weights))]
        quantization = self.config.quantization
        if quantization.type == 'deterministic':
            update = prob_clip.clip(update, quantization.bits, quantization.frac, False)
        elif quantization.type == 'probabilistic':
            update = prob_clip.clip(update, quantization.bits, quantization.frac, True)
        else:
            raise Exception('Selected quantization method does not exist!')

        return [old_weights[i] + update[i] for i in range(len(old_weights))]

    def perform_attack(self):
        try:
            attack_config = self.config.malicious
            from src import attack
            cls = getattr(attack, attack_config.objective["name"])
            attack: StealthAttack = cls()
            attack.set_stealth_method(self.evasion_factory())
        except Exception as e:
            raise ConfigurationError("Invalid attack configuration", e)

        args = attack_config.objective['args'].copy()
        args['loss_object'] = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        args['optimizer'] = self.build_optimizer(args)
        args.pop('learning_rate', None)
        args.pop('reduce_lr', None)
        args.pop('attacker_full_dataset', None)
        malicious_weights = attack.generate(self.dataset, self.model, **args)
        return malicious_weights

    def build_optimizer(self, optimizer_config):
        # return elaborate optimizer, potentially with stepdecay
        opt = optimizer_config['optimizer']
        lr = optimizer_config['learning_rate'] if 'learning_rate' in optimizer_config else None
        step_decay = optimizer_config['step_decay'] if 'step_decay' in optimizer_config else None
        if opt == "Adam":
            if lr is not None:
                if step_decay is not None and step_decay:
                    decay = StepDecay(lr,
                                      optimizer_config['num_epochs'] * optimizer_config['num_batch'])
                    optimizer_config['step_decay'] = decay
                    return tf.keras.optimizers.Adam(learning_rate=decay)
                return tf.keras.optimizers.Adam(learning_rate=lr)
        if opt == "SGD":
            if lr is not None:
                if step_decay is not None and step_decay:
                    decay = StepDecay(lr,
                                      optimizer_config['num_epochs'] * optimizer_config['num_batch'])
                    optimizer_config['step_decay'] = decay
                    return tf.keras.optimizers.SGD(learning_rate=decay)
                return tf.keras.optimizers.SGD(learning_rate=lr)
        return tf.keras.optimizers.Adam()


    def evasion_factory(self):
        """

        :rtype: EvasionMethod|None
        """
        attack_config = self.config.malicious
        if attack_config.evasion is None:
            return None
        evasion_name = attack_config.evasion['name']
        args = attack_config.evasion['args']
        from src.attack import evasion
        cls = getattr(evasion, evasion_name)
        if evasion_name == 'NormBoundPGDEvasion':
            return cls(old_weights=self.weights, benign_updates=self.benign_updates_this_round, **args)
        elif evasion_name == 'TrimmedMeanEvasion':
            assert self.benign_updates_this_round is not None, "Only full knowledge attack is supported at this moment"
            return cls(benign_updates_this_round=self.benign_updates_this_round, **args)
        else:
            raise NotImplementedError(f"Evasion with name {evasion_name} not supported.")

    @tf.function
    def optimized_training(self, batch_x, batch_y):
        """Uses tf non-eager execution using graph"""

        self.unoptimized_benign_training(batch_x, batch_y)

    def unoptimized_benign_training(self, batch_x, batch_y):
        with tf.GradientTape() as tape:
            predictions = self.model(batch_x, training=True)
            loss_value = self.loss_object(y_true=batch_y, y_pred=predictions)
            reg = tf.reduce_sum(self.model.losses)
            total_loss = loss_value + reg

        grads = self._compute_gradients_honest(tape, total_loss)
        self.honest_optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        self.train_loss(total_loss)
        self.train_accuracy(batch_y, predictions)

    def honest_training(self):
        """Performs local training"""
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=False)  # Our model has a softmax layer!

        num_iters = 0
        tboard_callback = tf.keras.callbacks.TensorBoard(log_dir='logdir',
                                                         histogram_freq=1,
                                                         profile_batch='10,30')
        if self.config.optimized_training:
            for i in range(self.config.benign_training.num_epochs):
                # tf.print(self.honest_optimizer._decayed_lr(tf.float32))
                for (batch_x, batch_y) in self.dataset.get_data():
                    self.optimized_training(batch_x, batch_y)
        else:
            self.honest_optimizer = self.create_honest_optimizer()

            for i in range(self.config.benign_training.num_epochs):
                for (batch_x, batch_y) in self.dataset.get_data():
                    # image_augmentation.debug(batch_x, batch_y)
                    self.unoptimized_benign_training(batch_x, batch_y)

            self.honest_optimizer = None  # release for memory reasons

        # Set last weights as we might act malicious next round
        if self.malicious:
            self.last_global_weights = self.weights  # First round, don't estimate
            self.last_update_weights = self.model.get_weights()

        return self.model.get_weights()

    def deviate_max_norm_attack(self):
        """Builds byzantine attack that has worst case opposite of gradient"""

        # Currently only l_inf is supported
        assert self.benign_updates_this_round is not None, "Only full knowledge attack is supported at this moment"

        aggregation = self.config['aggregator']
        if aggregation == "FedAvg":
            assert self.config["clip"] is not None

            # Full knowledge
            next_update = Client.average_weights(self.benign_updates_this_round)

            clip_value = self.config["clip"]
            new_weights_opposite_direction = [np.sign(layer) * -clip_value for layer in next_update]

            return new_weights_opposite_direction
        elif aggregation == "TrimmedMean":
            constant_b = 2.0

            accumulator = [np.zeros([*layer.shape, len(self.benign_updates_this_round)], layer.dtype)
                           for layer in self.benign_updates_this_round[0]]
            for client in range(0, len(self.benign_updates_this_round)):
                for layer in range(len(self.benign_updates_this_round[client])):
                    accumulator[layer][..., client] = self.benign_updates_this_round[client][layer]

            # put the per-client layers in single np array

            # next_update_check = Client.average_weights(self.benign_updates_this_round)
            next_update = [np.mean(layer, -1) for layer in accumulator]
            layer_max = [np.max(layer, -1) for layer in accumulator]
            layer_min = [np.min(layer, -1) for layer in accumulator]
            directions = [np.sign(layer) for layer in next_update]

            new_weights = []
            for signs, max, min in zip(directions, layer_max, layer_min):
                max_interval = np.where(max > 0, (max, max * constant_b), (max, max / constant_b))
                min_interval = np.where(min > 0, (min / constant_b, min), (constant_b * min, min))
                intervals = np.where(signs < 0, max_interval, min_interval)
                intervals = np.moveaxis(intervals, 0, -1)
                randomness = np.random.sample(intervals.shape[0:-1])
                weights = (intervals[..., 1] - intervals[..., 0]) * randomness + intervals[..., 0]
                new_weights.append(weights)

            return new_weights
        else:
            raise NotImplementedError("Aggregation method not supported by this attack.")

    def add_noise(self, batch_x):
        if self.config.gaussian_noise is None:
            return batch_x

        sigma = self.config.gaussian_noise
        gauss = np.random.normal(0, sigma, batch_x.shape)
        gauss = gauss.reshape(batch_x.shape).astype(batch_x.dtype)
        noisy = np.clip(batch_x + gauss, a_min=0.0, a_max=1.0)
        return noisy

    # def contamination_attack(self, optimizer, loss_object):
    #     """This attack modifies only epsilon*n neurons.
    #
    #     Inspired by: Diakonikolas, Ilias, et al. "Sever: A robust meta-algorithm for stochastic optimization. ICML 2019
    #
    #     Warning: the convergence parameter is hard-codded as 0.01
    #     """
    #     assert self.malicious and self.config['contamination_model']
    #
    #     # region contamination mask creation
    #     contamination_mask = [np.zeros_like(self.weights[i]) for i in range(len(self.weights))]
    #     layer_iter, layer_ind = 0, 0
    #     for ind in range(len(self.model.layers)):
    #         if type(self.model.layers[ind]) in [Conv2D, Dense]:
    #             elems = self.model.layers[ind].weights[0].shape[-1]
    #             elems_to_keep = int(elems * self.config['contamination_rate'][layer_iter])
    #             keep_inds = np.random.choice(elems, elems_to_keep, replace=False)
    #             contamination_mask[layer_ind][..., keep_inds] = 1  # weights
    #             contamination_mask[layer_ind + 1][keep_inds] = 1  # biases
    #
    #             layer_iter += 1
    #             layer_ind += 2
    #     # endregion
    #
    #     # backdoor with small noise
    #     batch_x = self.dataset.x_aux
    #     for local_epoch in range(100):  # maximum number of local epochs
    #         with tf.GradientTape() as tape:
    #             # add a small noise to data samples
    #             loss_value = loss_object(y_true=self.dataset.mal_aux_labels,
    #                                      y_pred=self.model(self.add_noise(batch_x), training=True))
    #             if loss_value < 0.01:
    #                 break
    #             grads = self._compute_gradients(tape, loss_value)
    #             # blackout gradients
    #             for i in range(len(grads)):
    #                 grads[i] = grads[i] * contamination_mask[i]
    #
    #             optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
    #
    #     # boost weights
    #     new_weights = self.apply_attack(self.weights, self.model.get_weights())
    #     return new_weights

    # def minimize_loss_attack(self, optimizer, loss_object, round):
    #     assert self.malicious
    #
    #     mal_optimizer = self.create_malicious_optimizer()
    #     # refine on auxiliary dataset
    #     loss_value = 100
    #     epoch = 0
    #     while epoch < self.config['mal_num_epochs'] or loss_value > self.config['mal_target_loss']:
    #         for mal_batch_x, mal_batch_y in self.dataset.get_aux(self.config['mal_num_batch']):
    #             with tf.GradientTape() as tape:
    #                 loss_value = loss_object(y_true=mal_batch_y,
    #                                          y_pred=self.model(self.add_noise(mal_batch_x), training=True))
    #                 # if loss_value > 0.01:
    #                 grads = self._compute_gradients(tape, loss_value)
    #                 mal_optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
    #
    #             self.update_weights_pgd()
    #
    #         # print(f"Loss value mal {loss_value}")
    #         epoch += 1
    #
    #         if epoch > self.config['mal_num_epochs_max']:
    #             logging.debug(f"Client {self.id}: Epoch break ({epoch})")
    #             break
    #
    #     # boost weights
    #     new_weights = self.apply_attack(self.weights, self.model.get_weights())
    #     return new_weights

    # def backdoor_stealth_attack(self, optimizer, loss_object, round):
    #     """Applies alternating minimization strategy, similar to bhagoji
    #
    #     First, we train the honest model for one batch, and then the malicious samples, if the loss is still high.
    #     Iterates for as many benign batches exist.
    #
    #     Note: Does not support the l2 norm constraint
    #     Note: Only supports boosting the full gradient, not only the malicious part.
    #     Note: Implemented as by Bhagoji. Trains the malicious samples with full batch size, this may not be desirable.
    #
    #     """
    #
    #     mal_optimizer = self.create_malicious_optimizer()
    #
    #     loss_value_malicious = 100
    #     epoch = 0
    #     # current_weights = self.model.get_weights()
    #     # delta_mal_local = [np.zeros(w.shape) for w in current_weights]
    #     while epoch < self.config['mal_num_epochs'] or loss_value_malicious > self.config['mal_target_loss']:
    #
    #         for (batch_x, batch_y) in self.dataset.get_data():
    #             with tf.GradientTape() as tape:
    #                 pred = self.model(batch_x, training=True)
    #                 pred_labels = np.argmax(pred, axis=1)
    #                 loss_value = loss_object(y_true=batch_y, y_pred=pred)
    #                 acc = np.mean(pred_labels == batch_y)
    #                 logging.debug(f"Client {self.id}: Benign loss {loss_value} {acc}")
    #                 grads = self._compute_gradients(tape, loss_value)
    #
    #                 optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
    #
    #             # self.update_weights_pgd()
    #
    #         # delta_benign = self.model.get_weights()
    #
    #         for (x_aux, y_aux) in self.dataset.get_aux(self.config['mal_num_batch']):
    #             pred_mal = self.model(x_aux, training=False)
    #             loss_value_malicious = loss_object(y_true=y_aux,
    #                                                y_pred=pred_mal)
    #             if loss_value_malicious < self.config['mal_target_loss']:
    #                 break
    #
    #             with tf.GradientTape() as tape:
    #                 pred_mal = self.model(x_aux, training=True)
    #                 pred_mal_labels = np.argmax(pred_mal, axis=1)
    #                 loss_value_malicious = loss_object(y_true=y_aux,
    #                                                    y_pred=pred_mal)
    #                 acc_mal = np.mean(pred_mal_labels == y_aux)
    #                 self.debug(f"Mal loss {loss_value_malicious} {acc_mal}")
    #                 grads = tape.gradient(loss_value_malicious, self.model.trainable_variables)
    #                 mal_optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
    #
    #         # end_weights = self.model.get_weights()
    #         # delta_mal = [delta_benign[i] - end_weights[i] for i in range(len(end_weights))]
    #         # delta_mal_local = [delta_mal_local[i] + delta_mal[i] for i in range(len(delta_mal))]
    #
    #         if epoch > self.config['mal_num_epochs_max']:
    #             self.debug(f"Epoch break! {loss_value_malicious}")
    #             break
    #
    #         epoch += 1
    #
    #     # end_weights = self.model.get_weights()
    #     # new_weights = [end_weights[i] + (delta_mal_local[i] * (self.config['scale_attack_weight'] - 1)) for i in range(len(delta_mal_local))]
    #     new_weights = self.apply_attack(self.weights, self.model.get_weights())
    #     return new_weights
    #
    # def model_replacement_attack(self, optimizer, loss_object, round):
    #     # Note: Implemented as by `Can you really backdoor federated learning?` baseline attack
    #
    #     poison_samples = self.config['poison_samples']
    #     mal_num_batch = self.config['mal_num_batch']
    #
    #     # Uses custom StepDecay because we want to step more explicitly
    #     if self.config['mal_step_learning_rate']:
    #         step_decay = StepDecay(self.config['mal_learning_rate'], self.config['mal_num_epochs'] * mal_num_batch)
    #         mal_optimizer = Model.create_optimizer(self.config["optimizer"], step_decay, None, None, None, None, None)
    #     else:
    #         step_decay = None
    #         mal_optimizer = Model.create_optimizer(self.config["optimizer"], self.config['mal_learning_rate'], None,
    #                                                None, None, None, None)
    #     # loss_object = regularized_loss(self.model.layers, self.weights)
    #
    #     loss_value_mal = 100
    #     for epoch in range(self.config['mal_num_epochs']):
    #         for batch_x, batch_y in self.dataset.get_data_with_aux(poison_samples, mal_num_batch):  # 10 is ICML
    #             print(f"LR: {mal_optimizer._decayed_lr(var_dtype=tf.float32)}")
    #             # image_augmentation.debug(batch_x, batch_y)
    #
    #             with tf.GradientTape() as tape:
    #                 loss_value = loss_object(y_true=batch_y, y_pred=self.model(batch_x, training=True))
    #                 # print(f"Loss: {loss_value}")
    #                 grads = self._compute_gradients(tape, loss_value)
    #                 mal_optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
    #
    #             if step_decay is not None:
    #                 step_decay.apply_step()
    #
    #             self.update_weights_pgd()
    #
    #             loss_value_mal, acc_mal = self.eval_aux_test(loss_object)
    #
    #             # acc_nonmal = tf.reduce_mean((batch_y[21:] == tf.argmax(self.model(batch_x[21:], training=True), axis=1)))
    #             # preds = self.model(batch_x[21:], training=False).numpy().argmax(axis=1)
    #             # pred_inds = preds == batch_y[21:].numpy()
    #             # # print(f"Correct: {self.global_dataset.y_aux_test[pred_inds]} -> {preds[pred_inds]}")
    #             # acc_nonmal = np.mean(pred_inds)
    #             logging.debug(f"Client {self.id}: Loss {loss_value_mal} acc {acc_mal}")
    #
    #             # if step_decay is not None:
    #             #     if loss_value_mal < self.config['mal_target_loss']:
    #             #         step_decay.mul = 0.01
    #             #     else:
    #             #         step_decay.mul = 1.0
    #
    #         # if loss_value_mal < self.config['mal_target_loss']:
    #         #     self.debug(f"Below target loss {loss_value_mal}")
    #         #     break
    #
    #         self.debug("Epoch")
    #
    #     new_weights = self.apply_attack(self.weights, self.model.get_weights())
    #     return new_weights

    def malicious_training(self, round):
        assert self.malicious

        optimizer = self.create_honest_optimizer()

        if self.config.benign_training.regularization_rate is not None:
            loss_object = regularized_loss(self.model.layers, self.weights, self.config.benign_training.regularization_rate)
        else:
            loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=False)  # Our model has a softmax layer!

        attack_type = self.attack_type

        logging.info(f"Client {self.id}: Malicious training")

        if attack_type == Attack.UNTARGETED:
            new_weights = self.perform_attack()
        elif attack_type == Attack.DEVIATE_MAX_NORM:
            new_weights = self.deviate_max_norm_attack()
        elif attack_type == Attack.BACKDOOR:
            new_weights = self.perform_attack()
        else:
            raise Exception('Unknown type of attack!')

        if self.config.malicious.estimate_other_updates:
            # I guess new_weights should be updated based on this difference
            new_weights = self.apply_estimation(self.weights, new_weights)

        return new_weights

    # @profile
    def train(self, round):
        """Performs local training"""

        self.train_loss.reset_states()
        self.train_accuracy.reset_states()
        self.pgd_step_counter = 0

        self.model.set_weights(self.weights)
        # self.global_trainable_weight = [w.numpy() for w in self.model.trainable_weights]

        if self.acting_malicious(round):
            new_weights = self.malicious_training(round)
        else:
            new_weights = self.honest_training()

        self.global_trainable_weight = None  # Release

        # loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        #     from_logits=False)  # Our model has a softmax layer!
        # self.eval_train(loss_object)

        new_weights = self.apply_defense(self.weights, new_weights)
        new_weights = self.apply_quantization(self.weights, new_weights)

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

        logging.info(f"Client {self.id}: Global model estimation")

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
        logging.info(f"Client {self.id}: Boosting weights with {self.config['scale_attack_weight']}")

        delta_weights = [(new_weights[i] - old_weights[i]) * self.config['scale_attack_weight']
                         for i in range(len(old_weights))]
        return [old_weights[i] + delta_weights[i] for i in range(len(old_weights))]

    def apply_defense(self, old_weights, new_weights):
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
        # clip_layers = self.config['clip_layers'] if self.config['clip_layers'] != [] else range(len(old_weights))
        clip_layers = range(len(old_weights))
        clip = self.config.clip
        if clip is None:
            return new_weights

        if clip.type == "linf":
            if clip.probability is None:
                delta_weights = [np.clip(delta_weights[i], -clip.value, clip.value) if i in clip_layers else delta_weights[i]
                                 for i in range(len(delta_weights))]

                # # Addition, clip layers less aggressively
                # delta_weights = [np.clip(delta_weights[i], -clip * 5, clip * 5) if i not in clip_layers else delta_weights[i]
                #                  for i in range(len(delta_weights))]
            else:
                delta_weights = self.random_clip_l0(delta_weights, clip.value, clip.probability, clip_layers)

        if clip.type == "l2":
            delta_weights = self.clip_l2(delta_weights, clip.value, clip_layers)

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
        norm = max(tf.norm(tf.concat(layers_to_clip, axis=0)), 0.00001)
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
        return self.malicious and self.config.malicious.attack_start <= round <= self.config.malicious.attack_stop

    # def apply_pgd_weights(self, old_weights, new_weights):
    #     pgd = self.config['pgd']
    #     if pgd is not None:
    #
    #         pgd_constraint = self.config['pgd_constraint'] / self.config['scale_attack_weight'] \
    #             if self.malicious and self.config['scale_attack'] \
    #             else self.config['pgd_constraint']
    #
    #         self.debug(f"Applying constraint {pgd} with value {pgd_constraint}")
    #
    #         if pgd == 'l_inf':
    #             new_weights = self.apply_defense(old_weights, new_weights, pgd_constraint, None)
    #         elif pgd == 'l2':
    #             new_weights = self.apply_defense(old_weights, new_weights, None, pgd_constraint)
    #         else:
    #             raise Exception('PGD type not supported')
    #     return new_weights

    # def update_weights_pgd(self):
    #     self.pgd_step_counter += 1
    #     if self.pgd_step_counter % self.config['pgd_clip_frequency'] != 0:
    #         # not yet time to clip
    #         return
    #
    #     new_weights = self.apply_pgd_weights(self.weights, self.model.get_weights())
    #     self.model.set_weights(new_weights)

    def eval_train(self, loss_object):
        total_loss = 0.0
        batch_count = 0.0
        for batch_x, batch_y in self.dataset.get_data():
            loss_value = loss_object(y_true=batch_y, y_pred=self.model(batch_x, training=False))

            total_loss += loss_value
            batch_count += 1

        loss = total_loss / batch_count
        logging.debug(f"Client {self.id}: Training loss {loss}")

    def eval_aux_test(self, loss_object):
        for batch_x, batch_y in self.dataset.get_aux_test_generator(1):
            preds = self.model(batch_x, training=False)
            loss_value = loss_object(y_true=batch_y, y_pred=preds)

            pred_inds = preds.numpy().argmax(axis=1) == batch_y
            adv_success = np.mean(pred_inds)

            return loss_value, adv_success


    def create_honest_optimizer(self):
        training = self.config.benign_training
        return Model.create_optimizer(training.optimizer, training.learning_rate,
                                      training.decay)

    def debug(self, v):
        logging.debug(f"Client {self.id}: {v}")

    def info(self, v):
        logging.info(f"Client {self.id}: {v}")

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
