from importlib import import_module

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Activation
from keras import backend as K, Input
import tensorflow as tf
import numpy as np

from fed_learning.server.global_model_config import GlobalModelConfig
from fed_learning.server.global_model import GlobalModel

from fed_learning.intrinsic_dimension.keras_ext.engine import ExtendedModel
from fed_learning.intrinsic_dimension.keras_ext.layers import (RProjDense,
                              RProjConv2D,
                              RProjBatchNormalization,
                              RProjLocallyConnected2D)
from fed_learning.intrinsic_dimension.keras_ext.rproj_layers_util import (OffsetCreatorDenseProj,
                                         OffsetCreatorSparseProj,
                                         OffsetCreatorFastfoodProj,
                                         FastWalshHadamardProjector,
                                         ThetaPrime)
from fed_learning.intrinsic_dimension.keras_ext.util import make_image_input_preproc, warn_misaligned_shapes
from keras.regularizers import l2

# NOTE mlei: for development purposes only, as it aims for shorter testing cycles
#            reduced NN width and depth
from fed_learning.intrinsic_dimension.general.tfutil import tf_get_uninitialized_variables, tf_assert_all_init, summarize_weights


def build_model(model_config: GlobalModelConfig, test_data):
    return DevFedMNISTCNNSubspace(model_config, test_data)

SUBSPACE_VSIZE = 1000
SEED = 41

class DevFedMNISTCNNSubspace(GlobalModel):
    
    def __init__(self, model_config: GlobalModelConfig, test_data):
        super(DevFedMNISTCNNSubspace, self).__init__(model_config, test_data)

    @classmethod
    def build_model(cls, seed=None):
        # ~less parameters
        weight_decay = 0.0001
        shift_in=None
        im_shape = (28, 28, 1)
        n_label_vals = 10
        im_dtype = 'float32'
        proj_type = "dense"
        vsize = SUBSPACE_VSIZE

        if seed is None:
            seed = SEED
        np.random.seed(seed)

        input_images, preproc_images = make_image_input_preproc(im_shape, im_dtype, shift_in=shift_in)
        input_labels = Input(batch_shape=(None,), dtype='int64')

        vv = ThetaPrime(vsize)

        # v2 = [v for v in tf.global_variables() if v.name == "Theta:0"][0]
        # print(f"V2 {v2}")

        xx = RProjConv2D(proj_type, 16, kernel_size=3, strides=1,  kernel_initializer='he_normal', padding='valid', activation='relu', kernel_regularizer=l2(weight_decay))(preproc_images)
        xx = RProjConv2D(proj_type, 8, kernel_size=3, strides=1,  kernel_initializer='he_normal', padding='valid', activation='relu', kernel_regularizer=l2(weight_decay))(xx)
        xx = MaxPooling2D((2, 2))(xx)
        xx = Flatten()(xx)
        xx = RProjDense(proj_type, 48, kernel_initializer='he_normal', activation='relu', kernel_regularizer=l2(weight_decay))(xx)
        probs = RProjDense(proj_type, 10, kernel_initializer='glorot_uniform', activation='softmax', kernel_regularizer=l2(weight_decay))(xx)
        model = ExtendedModel(inputs=input_images, outputs=probs, theta=vv)
        model.add_extra_trainable_weight(vv.var)

        cls.init_vars(model, proj_type)

        return model

    @classmethod
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

    def serialize_model_name(self):
        return "DevFedMNISTCNNSubspace"

    def get_all_weights(self):
        # return self.model.get_all_weights()
        return None

    def get_weights_seed(self):
        return SEED

    def params_count(self):
        return SUBSPACE_VSIZE
