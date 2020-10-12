import uuid
import codecs
import pickle

import keras
from keras import Model, optimizers
from keras import backend
import numpy as np
import tensorflow as tf

from fed_learning.client.util import prob_clip
from fed_learning.client.local_model_config import LocalModelConfig
from fed_learning.client.dataset import DataSet

import logging

from fed_learning.client.util.training import LocalModelTraining
from fed_learning.intrinsic_dimension.general.tfutil import tf_get_uninitialized_variables

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class LocalModel(object):

    def __init__(self, model_config: LocalModelConfig, dataset: DataSet):
        self.model_config = model_config
        self.dataset = dataset
        self.round_ids = []
        self.model = None
        self.training_helper = LocalModelTraining(self.dataset, self.model_config)
        logger.info('Initializing model with with dtype %s' % backend.floatx())

    #@run_native
    def set_model(self, model: Model):
        self.model = model
        self.training_helper.set_model(model)
        logger.info('Compiling model')
        # self.init_vars(model, "dense")
        self.model.compile(loss=self.model_config.loss,
                           optimizer=self.model_config.get_optimizer(),
                           metrics=self.model_config.metrics)
        self.model.summary(print_fn=logger.info)

    def set_all_weights(self, weights):
        """
        Optional. Set all weights including non trainable. Underlying model must support `set_all_weights` function
        :param weights: weights
        :return:
        """
        self.model.set_weights(weights)

    def setup_new_round(self, round_id, weights):
        self.server_weights = weights
        self.model.set_weights(weights)
        self.round_ids.append(round_id)
    
    def get_current_weights(self):
        return self.model.get_weights()

    def get_current_round_id(self):
        return self.round_ids[-1]

    def get_current_round_nr(self):
        return len(self.round_ids)

    #@run_native
    def train_one_round(self):
        # NOTE mlei: for now 1 validation and then submit
        logger.info('Training start')
        # self._verify_weights_norm()

        self.training_helper.train()
        self.evaluate()
        update = self._calculate_update()
        flattened_update = self._flatten_update(update)
        clipped = self._clip_if_needed([flattened_update])[0]
        # print(clipped.shape, np.sum(flattened_update))
        # print(clipped.shape, np.sum(clipped))
        return clipped

    def evaluate(self):
        score = self.model.evaluate(self.dataset.x_test, self.dataset.y_test, verbose=0)
        logger.info(
            '[EVAL] Test (round,loss,accuracy): (%d, %f, %f)' %
            (self.get_current_round_nr(), score[0], score[1]))
        return score

    def _flatten_update(self, update):
        return np.concatenate([x.ravel() for x in update])

    def _calculate_update(self):
        logger.info('Calculating update')
        new_weights = self.model.get_weights()
        update = []
        for old, new in zip(self.server_weights, new_weights):
            update.append(new - old)
        return update

    def _clip_if_needed(self, update):
        if not self.model_config.probabilistic_quantization:
            return update
        max_bits = min(self.model_config.fp_bits, self.model_config.range_bits)
        logger.info(f'Probabilistically clipping {len(update)} to {max_bits} bits')
        clipped = prob_clip.clip(update, max_bits, self.model_config.fp_frac) # Get the params?
        # [np.testing.assert_array_almost_equal(u.flatten(), c.flatten(), 2) for u,c in zip(update, clipped)]
        logger.info("Done clipping")
        return clipped

    def _verify_weights_norm(self):
        new_weights = self.model.get_all_weights()
        normish = np.sum([np.linalg.norm(l) for l in new_weights])
        print(f"Norm of weights: {normish}")

    def init_vars(self, model, proj_type):
        sess = keras.backend.get_session()
        uninitialized_vars = tf_get_uninitialized_variables(sess)
        init_missed_vars = tf.variables_initializer(uninitialized_vars, 'init_missed_vars')
        sess.run(init_missed_vars)

        # 3.5 Normalize the overall basis matrix across the (multiple) unnormalized basis matrices for each layer
        basis_matrices = []
        normalizers = []

        for layer in model.layers:
            try:
                basis_matrices.extend(layer.offset_creator.basis_matrices)
            except AttributeError:
                continue
            try:
                normalizers.extend(layer.offset_creator.basis_matrix_normalizers)
            except AttributeError:
                continue

        if len(basis_matrices) > 0: # and not args.load

            if proj_type == 'sparse':

                # Norm of overall basis matrix rows (num elements in each sum == total parameters in model)
                bm_row_norms = tf.sqrt(tf.add_n([tf.sparse_reduce_sum(tf.square(bm), 1) for bm in basis_matrices]))
                # Assign `normalizer` Variable to these row norms to achieve normalization of the basis matrix
                # in the TF computational graph
                rescale_basis_matrices = [tf.assign(var, tf.reshape(bm_row_norms,var.shape)) for var in normalizers]
                _ = sess.run(rescale_basis_matrices)
            elif proj_type == 'dense':
                bm_sums = [tf.reduce_sum(tf.square(bm), 1) for bm in basis_matrices]
                divisor = tf.expand_dims(tf.sqrt(tf.add_n(bm_sums)), 1)
                rescale_basis_matrices = [tf.assign(var, var / divisor) for var in basis_matrices]
                _ = sess.run(rescale_basis_matrices)
