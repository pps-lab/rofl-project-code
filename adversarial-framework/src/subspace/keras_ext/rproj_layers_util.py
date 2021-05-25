#! /usr/bin/env python

# Copyright (c) 2018 Uber Technologies, Inc.
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
# from keras.backend.tensorflow_backend import _convert_string_dtype
from tensorflow.keras import regularizers, constraints, initializers, activations
from IPython import embed
from sklearn.random_projection import SparseRandomProjection as SRP
from scipy.sparse import find
import time

import os
import sys

lab_root = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')
sys.path.insert(1, lab_root)


# from ops.fwh import fast_walsh_hadamard as c_fast_walsh_hadamard


###########
# 
# A quick fix for the following error
# from keras.backend.tensorflow_backend import _convert_string_dtype
# Keras 2.0.8 NameError: global name '_convert_string_dtype' is not defined
# Also called in rproj_layers.py

def _convert_string_dtype(dtype):
    if dtype == 'float16':
        return np.float16
    if dtype == 'float32':
        return np.float32
    elif dtype == 'float64':
        return np.float64
    elif dtype == 'int16':
        return np.int16
    elif dtype == 'int32':
        return np.int32
    elif dtype == 'int64':
        return np.int64
    elif dtype == 'uint8':
        return np.int8
    elif dtype == 'uint16':
        return np.uint16
    else:
        raise ValueError('Unsupported dtype:', dtype)


###########

class ThetaPrime(object):
    def __init__(self, size):
        # self.var = tf.Variable(np.random.randn(size).astype('float32'), trainable=True, name="ThetaPrime")
        # self.var = tf.Variable(np.zeros((size), dtype='float32'), trainable=True, name="ThetaPrime")
        self.var_2d = tf.Variable(np.zeros((1, size), dtype='float32'), trainable=True, name="ThetaPrime")
        self.size = size


# class ThetaPrimeLayer(Layer):
#     def __init__(self, size):
#         super(ThetaPrimeLayer, self).__init__()
#         self.size = size
#         self.var = None
#         self.var_2d = None
#
#     def build(self, input_shape):
#         self.var = tf.Variable(np.random.randn(self.size).astype('float32'), trainable=True, name="ThetaPrime")
#         # self.var = tf.Variable(np.zeros((size), dtype='float32'), trainable=True, name="ThetaPrime")
#         self.var_2d = tf.expand_dims(self.var, 0)

class MultiplyLayer(Layer):

    def __init__(self, var):
        super(MultiplyLayer, self).__init__()
        self.var = var

    def call(self, inputs, **kwargs):
        return inputs * self.var


###########
#
# OffsetCreator{Dense,Sparse,Fastfood}Proj
#
# These classes create offsets. Each layer is given a projector on
# construction and uses it as needed to create weight/bias/etc
# offsets.
#
###########

class OffsetCreateDenseProjExec():
    def __init__(self, weight_basis, ww, shape, name):
        self.weight_basis = weight_basis
        self.ww = ww
        self.shape = shape
        self.name = name

    def __call__(self, *args, **kwargs):
        return tf.reshape(tf.matmul(self.weight_basis.var_2d, self.ww, name=self.name), self.shape)


class OffsetCreatorDenseProj(object):
    def __init__(self):
        self.basis_matrices = []

    def create_theta_offset(self, weight_basis, shape, dtype, name=None):
        assert isinstance(weight_basis, ThetaPrime), 'weight_basis should be a ThetaPrime'

        if isinstance(shape, tf.TensorShape):
            shape = shape.as_list()

        # Create projection matrix ww
        total_dim = 1
        for dim in shape:
            assert dim is not None and dim > 0, 'dimensions must be known'
            total_dim *= dim
        seed = np.random.randint(10e8)

        ww_shape = (weight_basis.size, total_dim)
        ww_0 = np.random.normal(0.0, 1.0, size=ww_shape)
        ww = tf.Variable(ww_0, trainable=False, dtype=_convert_string_dtype(dtype), name='%s_ww' % name)

        return OffsetCreateDenseProjExec(weight_basis, ww, shape, name)

        # theta_offset = tf.reshape(tf.matmul(weight_basis.var_2d, ww, name="MatMully"), shape)
        #
        # self.basis_matrices.append(ww)
        #
        # return theta_offset, [ww]

class OffsetCreateSparseProjExec():
    def __init__(self, weight_basis, normalizer, ww, shape, name):
        self.weight_basis = weight_basis
        self.normalizer = normalizer
        self.ww = ww
        self.shape = shape
        self.name = name

    def __call__(self, *args, **kwargs):
        # Pre-multiply the normalizer by the low-rank parameter vector to avoid a sparse matrix - sparse matrix product,
        # which is not well-supported in Tensorflow (instead of theta_full = (P*N^-1)*theta_small where P*N^-1 is a row-normalized
        # projection matrix, do P*(N^-1*theta_small)). (N^-1*theta_small) can be written as simply an element-wise vector division.
        theta_small_norm = tf.divide(self.weight_basis.var_2d, self.normalizer)
        # theta_small_norm = self.weight_basis.var_2d

        # Compute delta from theta_0 using sparse projection
        # Note: sparse matrix must be first argument
        delta_theta_flat = tf.sparse.sparse_dense_matmul(self.ww, theta_small_norm, adjoint_a=True, adjoint_b=True)

        # Create theta
        theta_offset = tf.reshape(delta_theta_flat, self.shape)

        return theta_offset



class OffsetCreatorSparseProj(object):
    def __init__(self):
        self.basis_matrices = []
        self.basis_matrix_normalizers = []

    def create_theta_offset(self, weight_basis, shape, dtype, name=None):
        assert isinstance(weight_basis, ThetaPrime), 'weight_basis should be a ThetaPrime'

        if isinstance(shape, tf.TensorShape):
            shape = shape.as_list()

        # Create projection matrix ww
        total_dim = 1
        for dim in shape:
            assert dim is not None and dim > 0, 'dimensions must be known'
            total_dim *= dim

        # Generate location and relative scale of non zero elements
        M = SRP(weight_basis.size)._make_random_matrix(weight_basis.size, total_dim)
        fm = find(M)

        # Create sparse projection matrix from small vv to full theta space
        ww0 = tf.SparseTensor(indices=np.array([fm[0], fm[1]]).T, values=fm[2],
                              dense_shape=[weight_basis.size, total_dim])
        ww = tf.cast(ww0, _convert_string_dtype(dtype), name="SparseyCast")



        # Create diagonal normalization matrix that will be filled in when all layers are created, so that we can normalize each
        # row of the projection matrix (with length equal to the total number of parameters in the model) once we have all its elements.
        # This will hold the norms of the rows of the un-normalized projection matrix.
        norm = tf.sqrt(tf.sparse.reduce_sum(tf.square(ww)))

        # tf.sqrt(tf.add_n([tf.sparse_reduce_sum(tf.square(bm), 1) for bm in basis_matrices]))

        normalizer = tf.Variable(tf.tile([norm], [weight_basis.size]),
                                 trainable=False, name='%s_normalizer' % name)

        self.basis_matrices.append(ww)
        self.basis_matrix_normalizers.append(normalizer)

        return OffsetCreateSparseProjExec(weight_basis, normalizer, ww, shape, name)

        # # Pre-multiply the normalizer by the low-rank parameter vector to avoid a sparse matrix - sparse matrix product,
        # # which is not well-supported in Tensorflow (instead of theta_full = (P*N^-1)*theta_small where P*N^-1 is a row-normalized
        # # projection matrix, do P*(N^-1*theta_small)). (N^-1*theta_small) can be written as simply an element-wise vector division.
        # theta_small_norm = tf.divide(weight_basis.var_2d, normalizer)
        #
        # # Compute delta from theta_0 using sparse projection
        # # Note: sparse matrix must be first argument
        # delta_theta_flat = tf.sparse.sparse_dense_matmul(ww, theta_small_norm, adjoint_a=True, adjoint_b=True)
        #
        # # Create theta
        # theta_offset = tf.reshape(delta_theta_flat, shape)
        #
        # # ww0 = tf.sparse.to_dense(ww0, validate_indices=False, name="SparseyDense")
        # # # ww0 = tf.Variable(ww0, trainable=False)
        #
        # self.basis_matrices.append(ww)
        # self.basis_matrix_normalizers.append(normalizer)
        #
        # # Note: previous versions added only ww0 to _non_trainable_weights but skipped normalizer. Here we more correctly return both.
        # # return theta_offset, [ww0]
        # return theta_offset, [ww0, normalizer]

class OffsetCreateFastfoodProjExec():
    def __init__(self, weight_basis, normalizer, ww, shape, name):
        self.weight_basis = weight_basis
        self.normalizer = normalizer
        self.ww = ww
        self.shape = shape
        self.name = name

    def __call__(self, *args, **kwargs):
        # Pre-multiply the normalizer by the low-rank parameter vector to avoid a sparse matrix - sparse matrix product,
        # which is not well-supported in Tensorflow (instead of theta_full = (P*N^-1)*theta_small where P*N^-1 is a row-normalized
        # projection matrix, do P*(N^-1*theta_small)). (N^-1*theta_small) can be written as simply an element-wise vector division.
        theta_small_norm = tf.divide(self.weight_basis.var_2d, self.normalizer)
        # theta_small_norm = self.weight_basis.var_2d

        # Compute delta from theta_0 using sparse projection
        # Note: sparse matrix must be first argument
        delta_theta_flat = tf.sparse.sparse_dense_matmul(self.ww, theta_small_norm, adjoint_a=True, adjoint_b=True)

        # Create theta
        theta_offset = tf.reshape(delta_theta_flat, self.shape)

        return theta_offset


class OffsetCreatorFastfoodProj(object):
    def __init__(self):
        pass

    def create_theta_offset(self, weight_basis, shape, dtype, name=None):
        # Get offset from theta_0 (offset is initially 0)
        assert isinstance(weight_basis,
                          FastWalshHadamardProjector), 'weight_basis should be a FastWalshHadamardProjector instance'
        proj_tensor = weight_basis.get_projected_tensor(shape)
        return proj_tensor, []


###########
#
# FastWalshHadamardProjector
#
# This class is instantiated once per network and manages the whole
# projection from d to D.
#
###########

class FastWalshHadamardProjector(Layer):
    '''FastWalshHadamardProjector owns the d trainable parameters and
    generates the D projected parameters.

    FastWalshHadamardProjector must be instantiated before the model
    is built with d (known) and D (possibly hard to find before model
    is built). Thus some trickiness is necessary.
    '''

    def __init__(self, dd, DD, **kwargs):
        super(FastWalshHadamardProjector, self).__init__(**kwargs)
        self.dd = dd
        self.DD = DD
        self.index = 0
        self.d_vec = self.add_weight('d_vec', (self.dd,), initializer='zeros')
        self.project_vars, self.D_vec = tf_fastfood_transform(self.d_vec, self.dd, self.DD)
        for vv in self.project_vars:
            self._non_trainable_weights.append(vv)

    def get_projected_tensor(self, shape):
        if isinstance(shape, tf.TensorShape):
            shape = shape.as_list()
        total_size = np.prod(shape)
        assert self.index + total_size <= self.DD, 'Overrun D vector; requested too many projected tensors'
        # ret = self.D_vec[self.index:self.index + total_size]
        retflat = tf.slice(self.D_vec, [self.index], [total_size])
        # print 'D_vec is', self.D_vec, 'and ret is', retflat
        ret = tf.reshape(retflat, shape)
        # print '      ... now ret is', ret
        # print 'Sliced from %d to %d and reshaped to %s' % (self.index, total_size, repr(shape))
        self.index += total_size
        return ret

    def check_usage(self):
        if self.index == self.DD:
            print('FastWalshHadamardProjector usage is perfect: %d out of %d dimensions used' % (self.index, self.DD))
        else:
            raise Exception(
                'FastWalshHadamardProjector usage is off: %d out of %d dimensions used' % (self.index, self.DD))


###########
#
# Fast Walsh Hadamard functions
#
###########

def np_fast_walsh_hadamard(x, axis, normalize=True):
    '''Compute Fast Walsh-Hadamard transform in numpy.

    Args:
        x: tensor of shape (a0, a1, ... aN, L, b0, b1, ..., bN).
            L must be a power of two.

        axis: the "L" axis above, aka the axis over which to do the
            Hadamard transform. All other dimensions are left alone;
            data on those dimension do not interact.

        normalize: Whether to normalize the results such that applying
            the transform twice returns to the original input
            value. If True, return values are floats even if input was
            int.

    Returns:
        ret: transformed tensor with same shape as x


    Tests:

    Wikipedia case

    >>> x = np.array([1,0,1,0,0,1,1,0])

    >>> np_fast_walsh_hadamard(x, 0, False)
    array([ 4,  2,  0, -2,  0,  2,  0,  2])

    >>> np_fast_walsh_hadamard(np_fast_walsh_hadamard(x, 0), 0)
    array([ 1.,  0.,  1.,  0.,  0.,  1.,  1.,  0.])
    '''

    orig_shape = x.shape
    assert axis >= 0 and axis < len(orig_shape), (
            'For a vector of shape %s, axis must be in [0, %d] but it is %d'
            % (orig_shape, len(orig_shape) - 1, axis))
    h_dim = orig_shape[axis]
    h_dim_exp = int(round(np.log(h_dim) / np.log(2)))
    assert h_dim == 2 ** h_dim_exp, (
            'hadamard can only be computed over axis with size that is a power of two, but'
            ' chosen axis %d has size %d' % (axis, h_dim))
    working_shape_pre = [int(np.prod(orig_shape[:axis]))]  # prod of empty array is 1 :)
    working_shape_post = [int(np.prod(orig_shape[axis + 1:]))]  # prod of empty array is 1 :)
    working_shape_mid = [2] * h_dim_exp
    working_shape = working_shape_pre + working_shape_mid + working_shape_post
    # print 'working_shape is', working_shape
    ret = x.reshape(working_shape)

    for ii in range(h_dim_exp):
        dim = ii + 1
        arrs = np.split(ret, 2, axis=dim)
        assert len(arrs) == 2
        ret = np.concatenate((arrs[0] + arrs[1], arrs[0] - arrs[1]), axis=dim)

    if normalize:
        ret = ret / np.sqrt(float(h_dim))

    ret = ret.reshape(orig_shape)

    return ret


def _fast_walsh_hadamard_one_step(xx, axis):
    aa, bb = tf.split(xx, 2, axis=axis)
    ret = tf.concat((aa + bb, aa - bb), axis=axis)
    return ret


def _fast_walsh_hadamard_one_step_method2(xx, pre, d1, d2, d3, post):
    working_shape = tf.concat((pre, d1, d2, d3, post), axis=0)
    xx = tf.reshape(xx, working_shape)
    aa, bb = tf.split(xx, 2, axis=2)
    ret = tf.concat((aa + bb, aa - bb), axis=2)
    return ret


def tf_fast_walsh_hadamard(in_x, axis, normalize=True, method='two'):
    '''Compute Fast Walsh-Hadamard transform in tensorflow.

    Args:
        x: tensor of shape (a0, a1, ... aN, L, b0, b1, ..., bN).
            L must be a power of two.

        axis: the "L" axis above, aka the axis over which to do the
            Hadamard transform. All other dimensions are left alone;
            data on those dimension do not interact.

        normalize: Whether to normalize the results such that applying
            the transform twice returns to the original input
            value.

        method:
            'one': Original reshape to [2]*ll version
            'two': Deal with TF "UnimplementedError: SliceOp : Unhandled input dimensions" error...
            'c':   Use C++ FWH Op.

    Returns:
        ret: transformed tensor with same shape as x. Returned
            tensor is always float even if input was int.


    Tests:

    >>> in_x = tf.placeholder('float32')
    >>> in_x
    <tf.Tensor 'Placeholder:0' shape=<unknown> dtype=float32>
    >>> sess = tf.InteractiveSession()


    Wikipedia case:

    >>> x = np.array([1,0,1,0,0,1,1,0])

    >>> sess.run(tf_fast_walsh_hadamard(in_x, 0, False), feed_dict={in_x: x})
    array([ 4.,  2.,  0., -2.,  0.,  2.,  0.,  2.], dtype=float32)

    >>> sess.run(tf_fast_walsh_hadamard(in_x, 0, False, method='two'), feed_dict={in_x: x})
    array([ 4.,  2.,  0., -2.,  0.,  2.,  0.,  2.], dtype=float32)

    >>> sess.run(tf_fast_walsh_hadamard(tf_fast_walsh_hadamard(in_x, 0), 0), feed_dict={in_x: x})
    array([ 1.,  0.,  1.,  0.,  0.,  1.,  1.,  0.], dtype=float32)


    Verify equivalence with numpy approach:

    >>> np.random.seed(123)
    >>> x = np.random.uniform(0, 1, (3, 64, 5))

    >>> h_np = np_fast_walsh_hadamard(x, 1)
    >>> h_tf_ = tf_fast_walsh_hadamard(in_x, 1)
    >>> h_tf2_ = tf_fast_walsh_hadamard(in_x, 1, method='two')
    >>> h_tf = sess.run(h_tf_, feed_dict={in_x: x})
    >>> h_tf2 = sess.run(h_tf2_, feed_dict={in_x: x})

    >>> x.shape
    (3, 64, 5)
    >>> h_np.shape
    (3, 64, 5)
    >>> h_tf.shape
    (3, 64, 5)
    >>> h_tf2.shape
    (3, 64, 5)

    >>> abs(h_np - h_tf).max() < 1e-6
    True
    >>> abs(h_np - h_tf2).max() < 1e-6
    True

    Try a few other shapes / axes

    >>> sess.run(tf_fast_walsh_hadamard(in_x, 0), feed_dict={in_x: x[0]}).shape == x[0].shape
    True
    >>> sess.run(tf_fast_walsh_hadamard(in_x, 1), feed_dict={in_x: x[:, :, 0]}).shape == x[:, :, 0].shape
    True
    >>> sess.run(tf_fast_walsh_hadamard(in_x, 0), feed_dict={in_x: x[0, :, 0]}).shape == x[0, :, 0].shape
    True
    '''

    orig_shape = tf.shape(input=in_x)
    h_dim = orig_shape[axis]
    h_dim_exp = tf.cast(tf.round(tf.math.log(tf.cast(h_dim, dtype=tf.float32)) / np.log(2)), 'int32')

    assert_pow2 = tf.compat.v1.assert_equal(h_dim, tf.pow(2, h_dim_exp),
                                            message='hadamard can only be computed over axis with size that is a power of two')

    with tf.control_dependencies([assert_pow2]):
        working_shape_pre = tf.expand_dims(tf.reduce_prod(input_tensor=orig_shape[:axis]),
                                           axis=0)  # reduce_prod of empty array is 1
        working_shape_post = tf.expand_dims(tf.reduce_prod(input_tensor=orig_shape[axis + 1:]),
                                            axis=0)  # reduce_prod of empty array is 1

    ii = tf.constant(0)
    assert method in ('one', 'two', 'c')
    if method == 'one':
        # expand to working dims [pre, 2, 2, 2, ..., 2, 2, post]
        working_shape_mid = tf.tile([2], [h_dim_exp])

        working_shape = tf.concat((working_shape_pre, working_shape_mid, working_shape_post),
                                  axis=0)

        ret_0 = tf.reshape(in_x, working_shape)

        cond = lambda i, x: tf.less(i, h_dim_exp)
        body = lambda i, x: (tf.add(i, 1), _fast_walsh_hadamard_one_step(x, i + 1))

        ii_final, ret = tf.while_loop(
            cond=cond,
            body=body,
            loop_vars=[ii, ret_0],
            parallel_iterations=1  # check on this?
        )
    elif method == 'two':
        # Never expand to high rank. Roll dimensions instead. This is
        # needed because backprop through the slice operator only
        # supports up to rank 7 tensors in TF 1.3
        # [pre, 1, 2, h_dim/2, post] ->
        # [pre, 2, 2, h_dim/4, post] -> ...
        # [pre, h_dim/2, 2, 1, post]

        d1 = tf.expand_dims(tf.constant(1), axis=0)
        d2 = tf.expand_dims(tf.constant(2), axis=0)  # always 2
        d3 = tf.expand_dims(tf.math.floordiv(h_dim, 2), axis=0)

        working_shape_0 = tf.concat((working_shape_pre, d1, d2, d3, working_shape_post), axis=0)
        ret_0 = tf.reshape(in_x, working_shape_0)

        cond = lambda i, d1, d3, x: tf.less(i, h_dim_exp)
        body = lambda i, d1, d3, x: (tf.add(i, 1),
                                     d1 * 2,
                                     tf.math.floordiv(d3, 2),
                                     _fast_walsh_hadamard_one_step_method2(x, working_shape_pre, d1, d2, d3,
                                                                           working_shape_post))

        ii_final, d1_final, d3_final, ret = tf.while_loop(
            cond=cond,
            body=body,
            loop_vars=[ii, d1, d3, ret_0],
            parallel_iterations=1  # check on this?
        )
    else:
        # 'c' version
        # Only works for rank-1 (vector) input

        assert False, 'c version disabled for now'

        assert axis == 0, 'axis must be 0 for the c version of tf_fast_walsh_hadamard'
        assert normalize, 'for c version normalize must be True'
        assert_rank1 = tf.compat.v1.assert_rank(in_x, 1)

        with tf.control_dependencies([assert_rank1, assert_pow2]):
            ret = c_fast_walsh_hadamard(in_x)

    if normalize and method != 'c':
        ret = ret / tf.sqrt(tf.cast(h_dim, dtype=tf.float32))

    ret = tf.reshape(ret, orig_shape)

    return ret


def tf_fastfood_transform(in_x, dd, DD, use_get=False, use_C=False):
    '''Transform from d to D. Pads as necessary.

    For now: assume dd and DD are known in python.'''

    # Tensor d and D
    # assert_D_big = tf.assert_greater_equal(DD, dd, message='d cannot be larger than D')
    # with tf.control_dependencies([assert_D_big]):
    #    ll = tf.cast(tf.round(tf.log(tf.to_float(DD)) / np.log(2)), 'int32')
    #    LL = tf.pow(2, ll)

    # Python d and D
    assert isinstance(dd, int), 'd should be int'
    assert isinstance(DD, int), 'D should be int'
    assert DD >= dd, 'd cannot be larger than D'
    assert dd > 0, 'd and D must be positive'

    ll = int(np.ceil(np.log(DD) / np.log(2)))
    LL = 2 ** ll

    # Make vars
    init_BB = tf.cast(tf.random.uniform((LL,), 0, 2, dtype='int32'), dtype=tf.float32) * 2 - 1
    init_Pi = tf.random.shuffle(tf.range(LL))
    init_GG = tf.random.normal((LL,))
    init_divisor = lambda GG: tf.sqrt(LL * tf.reduce_sum(input_tensor=tf.pow(GG.initialized_value(), 2)))
    if use_get:
        BB = tf.compat.v1.get_variable('B', initializer=init_BB, trainable=False)
        Pi = tf.compat.v1.get_variable('Pi', initializer=init_Pi, trainable=False)
        GG = tf.compat.v1.get_variable('G', initializer=init_GG, trainable=False)
        divisor = tf.compat.v1.get_variable('divisor', initializer=init_divisor(GG), trainable=False)
    else:
        BB = tf.Variable(init_BB, name='B', trainable=False)
        Pi = tf.Variable(init_Pi, name='Pi', trainable=False)
        GG = tf.Variable(init_GG, name='G', trainable=False)
        divisor = tf.Variable(init_divisor(GG), name='divisor', trainable=False)

    fastfood_vars = [BB, Pi, GG, divisor]

    return fastfood_vars, ret

class FastfoodExec():
    def __init__(self, in_x, BB, Pi, LL, ll, DD, dd, divisor, use_C):
        self.in_x = in_x
        self.BB = BB
        self.Pi = Pi
        self.LL = LL
        self.ll = ll
        self.DD = DD
        self.dd = dd
        self.divisor = divisor
        self.use_C = use_C

    def __call__(self, *args, **kwargs):

        # Implement transform
        dd_pad = tf.pad(tensor=self.in_x, paddings=[[0, self.LL - self.dd]])
        mul_1 = tf.multiply(self.BB, dd_pad)
        if self.use_C:
            mul_2 = tf_fast_walsh_hadamard(mul_1, 0, method='c', normalize=True)
        else:
            mul_2 = tf_fast_walsh_hadamard(mul_1, 0, method='two', normalize=False)
        mul_3 = tf.gather(mul_2, self.Pi)
        mul_4 = tf.multiply(mul_3, self.GG)
        if self.use_C:
            mul_5 = tf_fast_walsh_hadamard(mul_4, 0, method='c', normalize=True)
            print('\nWARNING: check normalization on this next line more carefully\n')
            ret = tf.divide(tf.slice(mul_5, [0], [self.DD]), self.divisor * np.sqrt(float(self.DD) / self.LL / self.ll))
        else:
            mul_5 = tf_fast_walsh_hadamard(mul_4, 0, method='two', normalize=False)
            ret = tf.divide(tf.slice(mul_5, [0], [self.DD]), self.divisor * np.sqrt(float(self.DD) / self.LL))


def test_timing():
    N = 29

    in_x = tf.compat.v1.placeholder('float32')
    sum_x = tf.reduce_sum(input_tensor=in_x)
    hh = tf_fast_walsh_hadamard(in_x, 1, True)
    sum_h = tf.reduce_sum(input_tensor=hh)
    sess = tf.compat.v1.InteractiveSession()

    for ll in range(1, N):
        L = 2 ** ll
        print('\n%d, H dim %d' % (ll, L))

        x = np.random.uniform(0, 1, (1, L, 1))

        if L < 33554432:
            start = time.time()
            np_fast_walsh_hadamard(x, 1)
            end = time.time()
            print('  np                %14s elems:  %16s' % ('%d' % L, '%f' % (end - start)))
        else:
            print('  np                     <skipped>')

        start = time.time()
        sess.run(sum_h, feed_dict={in_x: x})
        end = time.time()
        print('  tf                %14s elems:  %16s' % ('%d' % L, '%f' % (end - start)))

        # Time each op the third time (ignore CUDA tuning time) then subtract data transfer time
        sess.run(sum_x, feed_dict={in_x: x})
        sess.run(sum_x, feed_dict={in_x: x})
        start = time.time()
        sess.run(sum_x, feed_dict={in_x: x})
        elap_data = time.time() - start
        sess.run(sum_h, feed_dict={in_x: x})
        sess.run(sum_h, feed_dict={in_x: x})
        start = time.time()
        sess.run(sum_h, feed_dict={in_x: x})
        elap_had = time.time() - start
        print('  tf just H         %14s elems:  %16s' % ('%d' % (L), '%f' % (elap_had - elap_data)))

        DD = max(5, int(np.ceil(L * .8)))
        dd = max(3, int(np.ceil(DD * .001)))
        if x.shape[1] >= dd:
            for use_C in [False, True]:
                st = '(C) ' if use_C else '(TF)'
                ffvars, xform = tf_fastfood_transform(in_x, dd, DD, use_C=use_C)
                sum_xf = tf.reduce_sum(input_tensor=xform)
                sess.run(tf.compat.v1.global_variables_initializer())

                sess.run(sum_xf, feed_dict={in_x: x[0, :dd, 0]})
                start = time.time()
                sess.run(sum_xf, feed_dict={in_x: x[0, :dd, 0]})
                end = time.time()
                print('  tf %s fastf     %14s elems:  %16s' % (st, '%d' % L, '%f' % (end - start)))

                sess.run(sum_x, feed_dict={in_x: x[0, :dd, 0]})
                sess.run(sum_x, feed_dict={in_x: x[0, :dd, 0]})
                start = time.time()
                sess.run(sum_x, feed_dict={in_x: x[0, :dd, 0]})
                elap_data = time.time() - start
                sess.run(sum_xf, feed_dict={in_x: x[0, :dd, 0]})
                sess.run(sum_xf, feed_dict={in_x: x[0, :dd, 0]})
                start = time.time()
                sess.run(sum_xf, feed_dict={in_x: x[0, :dd, 0]})
                elap_had = time.time() - start
                print('  tf %s just fastf%14s elems:  %16s' % (st, '%d' % (L), '%f' % (elap_had - elap_data)))

        else:
            print('  tf fastfood       %14s elems:  <skipped, too small>' % ('%d' % L))

        if L > 32768:
            print('                        <skipped large batch cases>')
            continue

        x2 = np.random.uniform(0, 1, (10, L, 100))

        start = time.time()
        np_fast_walsh_hadamard(x2, 1)
        end = time.time()
        print('  np                %14s elems:  %16s' % ('%d' % (L * 1000), '%f' % (end - start)))

        start = time.time()
        sess.run(sum_h, feed_dict={in_x: x2})
        end = time.time()
        print('  tf                %14s elems:  %16s' % ('%d' % (L * 1000), '%f' % (end - start)))

        # Time each op the third time (ignore CUDA tuning time) then subtract data transfer time
        sess.run(sum_x, feed_dict={in_x: x2})
        sess.run(sum_x, feed_dict={in_x: x2})
        start = time.time()
        sess.run(sum_x, feed_dict={in_x: x2})
        elap_data = time.time() - start
        sess.run(sum_h, feed_dict={in_x: x2})
        sess.run(sum_h, feed_dict={in_x: x2})
        start = time.time()
        sess.run(sum_h, feed_dict={in_x: x2})
        elap_had = time.time() - start
        print('  tf just H         %14s elems:  %16s' % ('%d' % (L * 1000), '%f' % (elap_had - elap_data)))

    print('The next dim, 2**29 ==', 2 ** 29, 'crashes with OOM on a TitanX')


if __name__ == '__main__':
    import doctest

    doctest.testmod()

    test_timing()
