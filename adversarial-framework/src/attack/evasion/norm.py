

from .evasion_method import EvasionMethod
import logging
import numpy as np
import tensorflow as tf

class NormBoundPGDEvasion(EvasionMethod):
    """
    Evades norm bound using PGD.
    """

    def __init__(self, old_weights, norm_type, scale_factor, clipping_bound=None, pgd_factor=None):
        """

        :param old_weights: current weights
        :param norm_type: l2 or linf
        :param clipping_bound: clipping bound value
        :param scale_factor: factor with which to scale the update
        :param pgd_factor: factor with which to apply pgd
        """
        super().__init__(alpha=None) # Alpha is 0 because we use PGD
        self.weights = old_weights
        self.norm_type = norm_type
        self.clipping_bound = clipping_bound
        self.scale_factor = scale_factor
        self.pgd_factor = pgd_factor

    def update_after_batch(self, model):
        if self.pgd_factor is not None:
            # clip pgd
            new_weights = model.get_weights()
            new_weights = self.apply_pgd_weights(self.weights, new_weights)
            model.set_weights(new_weights)

    def update_after_training(self, model):
        # scale!
        new_weights = model.get_weights()
        delta_weights = [(new_weights[i] - self.weights[i]) * self.scale_factor
                         for i in range(len(self.weights))]
        update = [self.weights[i] + delta_weights[i] for i in range(len(self.weights))]
        model.set_weights(update)

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
        # clip_layers = self.config['clip_layers'] if self.config['clip_layers'] != [] else range(len(old_weights))
        clip_layers = range(len(old_weights))

        if clip is not None:
            delta_weights = [np.clip(delta_weights[i], -clip, clip) if i in clip_layers else delta_weights[i]
                             for i in range(len(delta_weights))]

        if clip_l2 and clip_l2 > 0:
            delta_weights = self.clip_l2(delta_weights, clip_l2, clip_layers)

        new_weights = [old_weights[i] + delta_weights[i] for i in range(len(old_weights))]

        return new_weights

    def apply_pgd_weights(self, old_weights, new_weights):
        pgd = self.norm_type
        if self.pgd_factor is not None:

            pgd_constraint = self.pgd_factor

            self.debug(f"Applying constraint {pgd} with value {pgd_constraint}")

            if pgd == 'linf':
                new_weights = self.apply_defense(old_weights, new_weights, pgd_constraint, None)
            elif pgd == 'l2':
                new_weights = self.apply_defense(old_weights, new_weights, None, pgd_constraint)
            else:
                raise Exception('PGD type not supported')
        return new_weights

    def clip_l2(self, delta_weights, l2, clip_layers):
        """
        Calculates the norm per layer.
        :param delta_weights: current weight update
        :param l2: l2 bound
        :param clip_layers: what layers to apply clipping to
        :return:
        """

        l2_norm_tensor = tf.constant(l2, dtype=tf.float32)
        layers_to_clip = [tf.reshape(delta_weights[i], [-1]) for i in range(len(delta_weights)) if
                          i in clip_layers]  # for norm calculation
        norm = tf.norm(tf.concat(layers_to_clip, axis=0))

        multiply = min((l2_norm_tensor / norm).numpy(), 1.0)

        return [delta_weights[i] * multiply if i in clip_layers else delta_weights[i] for i in
                range(len(delta_weights))]

    def debug(self, v):
        logging.debug(f"Norm bound stealth: {v}")