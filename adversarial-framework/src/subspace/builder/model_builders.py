
import numpy as np
from IPython import embed
import pdb

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (Dense, Flatten, Input, Activation,
                          Reshape, Dropout, Convolution2D,
                          MaxPooling2D, BatchNormalization,
                          Conv2D, GlobalAveragePooling2D,
                          Concatenate, AveragePooling2D,
                          LocallyConnected2D)
import keras.backend as K

# from general.tfutil import hist_summaries_traintest, scalar_summaries_traintest

from src.subspace.keras_ext.engine import ExtendedModel
from src.subspace.keras_ext.layers import (RProjDense,
                                       RProjConv2D,
                                       RProjBatchNormalization,
                                       RProjLocallyConnected2D)
from src.subspace.keras_ext.rproj_layers_util import (OffsetCreatorDenseProj,
                                                  OffsetCreatorSparseProj,
                                                  OffsetCreatorFastfoodProj,
                                                  FastWalshHadamardProjector,
                                                  ThetaPrime, MultiplyLayer)
from src.subspace.keras_ext.util import make_image_input_preproc
from tensorflow.keras.regularizers import l2


def make_and_add_losses(model, input_labels):
    '''Add classification and L2 losses'''

    with tf.compat.v1.name_scope('losses') as scope:
        prob = tf.nn.softmax(model.v.logits, name='prob')
        cross_ent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model.v.logits, labels=input_labels, name='cross_ent')
        loss_cross_ent = tf.reduce_mean(input_tensor=cross_ent, name='loss_cross_ent')
        model.add_trackable('loss_cross_ent', loss_cross_ent)
        class_prediction = tf.argmax(input=prob, axis=1)

        prediction_correct = tf.equal(class_prediction, input_labels, name='prediction_correct')
        accuracy = tf.reduce_mean(input_tensor=tf.cast(prediction_correct, dtype=tf.float32), name='accuracy')
        model.add_trackable('accuracy', accuracy)
        # hist_summaries_traintest(prob, cross_ent)
        # scalar_summaries_traintest(accuracy)

        model.add_loss_reg()
        if 'loss_reg' in model.v:
            loss = tf.add_n((
                model.v.loss_cross_ent,
                model.v.loss_reg,
            ), name='loss')
        else:
            loss = model.v.loss_cross_ent
        model.add_trackable('loss', loss)

    nontrackable_fields = ['prob', 'cross_ent', 'class_prediction', 'prediction_correct']
    for field in nontrackable_fields:
        model.add_var(field, locals()[field])



def build_model_mnist_fc(weight_decay=0, vsize=100, depth=2, width=100, shift_in=None, proj_type='dense'):
    im_shape = (28, 28, 1)
    n_label_vals = 10
    im_dtype = 'float32'

    assert proj_type in ('dense', 'sparse')
    if proj_type == 'dense':
        offset_creator_class = OffsetCreatorDenseProj
    else:
        # sparse
        offset_creator_class = OffsetCreatorSparseProj

    with tf.compat.v1.name_scope('inputs'):
        input_images, preproc_images = make_image_input_preproc(im_shape, im_dtype, shift_in=shift_in)
        input_labels = Input(batch_shape=(None,), dtype='int64')

    with tf.compat.v1.name_scope('net') as scope:
        vv = ThetaPrime(vsize)

        xx = input_images
        xx = Flatten()(xx)
        for _ in range(depth):
            xx = RProjDense(offset_creator_class, vv, width, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(xx)
            # xx = Dense(width, activation='relu')(xx)

        logits = RProjDense(offset_creator_class, vv, 10, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), activation='softmax')(xx)
        # model = Model(input=input_images, output=logits)
        model = ExtendedModel(input=input_images, output=logits)

        model.add_extra_trainable_weight(vv.var_2d)

    nontrackable_fields = ['input_images', 'preproc_images', 'input_labels', 'logits']
    for field in nontrackable_fields:
        model.add_var(field, locals()[field])

    make_and_add_losses(model, input_labels)

    return model, vv.var_2d

def build_cnn_model_mnist_bhagoji(weight_decay=0, vsize=100, shift_in=None, proj_type='sparse'):
    im_shape = (28, 28, 1)
    n_label_vals = 10
    im_dtype = 'float32'

    if proj_type == 'dense':
        offset_creator_class = OffsetCreatorDenseProj
    else:
        # sparse
        offset_creator_class = OffsetCreatorSparseProj

    with tf.name_scope('inputs'):
        input_images, preproc_images = make_image_input_preproc(im_shape, im_dtype, shift_in=shift_in)
        input_labels = Input(batch_shape=(None,), dtype='int64')

    with tf.name_scope('net') as scope:
        vv = ThetaPrime(vsize)

        xx = RProjConv2D(offset_creator_class, vv, 64, kernel_size=5, strides=1,  kernel_initializer='he_normal', padding='valid', activation='relu', kernel_regularizer=l2(weight_decay))(preproc_images)
        xx = RProjConv2D(offset_creator_class, vv, 64,  kernel_size=5, strides=1,  kernel_initializer='he_normal', padding='valid', activation='relu', kernel_regularizer=l2(weight_decay))(xx)
        # xx = MaxPooling2D((2, 2))(xx)
        xx = Flatten()(xx)
        xx = RProjDense(offset_creator_class, vv, 128, kernel_initializer='he_normal', activation='relu',
                        kernel_regularizer=l2(weight_decay))(xx)

        logits = RProjDense(offset_creator_class, vv, 10, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), activation='softmax')(xx)
        model = ExtendedModel(input=input_images, output=logits)

        model.add_extra_trainable_weight(vv.var_2d)

    nontrackable_fields = ['input_images', 'preproc_images', 'input_labels', 'logits']
    for field in nontrackable_fields:
        model.add_var(field, locals()[field])

    make_and_add_losses(model, input_labels)

    return model

def build_cnn_model_mnist_dev_conv(weight_decay=0, vsize=100, shift_in=None, proj_type='sparse'):
    im_shape = (28, 28, 1)
    n_label_vals = 10
    im_dtype = 'float32'

    if proj_type == 'dense':
        offset_creator_class = OffsetCreatorDenseProj
    else:
        # sparse
        offset_creator_class = OffsetCreatorSparseProj

    with tf.name_scope('inputs'):
        input_images, preproc_images = make_image_input_preproc(im_shape, im_dtype, shift_in=shift_in)
        input_labels = Input(batch_shape=(None,), dtype='int64')

    with tf.name_scope('net') as scope:
        vv = ThetaPrime(vsize)

        xx = RProjConv2D(offset_creator_class, vv, 8, kernel_size=3, strides=1,  kernel_initializer='he_normal', padding='valid', activation='relu', kernel_regularizer=l2(weight_decay))(preproc_images)
        xx = RProjConv2D(offset_creator_class, vv, 4,  kernel_size=3, strides=1,  kernel_initializer='he_normal', padding='valid', activation='relu', kernel_regularizer=l2(weight_decay))(xx)
        xx = MaxPooling2D((2, 2))(xx)
        xx = Flatten()(xx)
        xx = RProjDense(offset_creator_class, vv, 32, kernel_initializer='he_normal', activation='relu',
                        kernel_regularizer=l2(weight_decay))(xx)

        logits = RProjDense(offset_creator_class, vv, 10, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), activation='softmax')(xx)
        model = ExtendedModel(input=input_images, output=logits)

        model.add_extra_trainable_weight(vv.var_2d)

    nontrackable_fields = ['input_images', 'preproc_images', 'input_labels', 'logits']
    for field in nontrackable_fields:
        model.add_var(field, locals()[field])

    make_and_add_losses(model, input_labels)

    return model

def build_cnn_model_mnistcnn_conv(weight_decay=0, vsize=100, shift_in=None, proj_type='sparse'):
    im_shape = (28, 28, 1)
    n_label_vals = 10
    im_dtype = 'float32'

    if proj_type == 'dense':
        offset_creator_class = OffsetCreatorDenseProj
    else:
        # sparse
        offset_creator_class = OffsetCreatorSparseProj

    with tf.name_scope('inputs'):
        input_images, preproc_images = make_image_input_preproc(im_shape, im_dtype, shift_in=shift_in)
        input_labels = Input(batch_shape=(None,), dtype='int64')

    with tf.name_scope('net') as scope:
        vv = ThetaPrime(vsize)

        xx = RProjConv2D(offset_creator_class, vv, 64, kernel_size=2, strides=1,  kernel_initializer='he_normal', padding='valid', activation='relu', kernel_regularizer=l2(weight_decay))(preproc_images)
        xx = MaxPooling2D((2, 2))(xx)
        xx = RProjConv2D(offset_creator_class, vv, 32,  kernel_size=2, strides=1,  kernel_initializer='he_normal', padding='valid', activation='relu', kernel_regularizer=l2(weight_decay))(xx)
        xx = MaxPooling2D((2, 2))(xx)
        xx = Flatten()(xx)
        xx = RProjDense(offset_creator_class, vv, 256, kernel_initializer='he_normal', activation='relu',
                        kernel_regularizer=l2(weight_decay))(xx)

        logits = RProjDense(offset_creator_class, vv, 10, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), activation='softmax')(xx)
        model = ExtendedModel(input=input_images, output=logits)

        model.add_extra_trainable_weight(vv.var_2d)

    nontrackable_fields = ['input_images', 'preproc_images', 'input_labels', 'logits']
    for field in nontrackable_fields:
        model.add_var(field, locals()[field])

    make_and_add_losses(model, input_labels)

    return model

def build_cnn_model_cifar_allcnn(weight_decay=0, vsize=100, shift_in=None, proj_type='sparse'):
    im_shape = (32, 32, 3)
    n_label_vals = 10
    im_dtype = 'float32'

    if proj_type == 'dense':
        offset_creator_class = OffsetCreatorDenseProj
    else:
        # sparse
        offset_creator_class = OffsetCreatorSparseProj

    with tf.name_scope('inputs'):
        input_images, preproc_images = make_image_input_preproc(im_shape, im_dtype, shift_in=shift_in)
        input_labels = Input(batch_shape=(None,), dtype='int64')

    with tf.name_scope('net') as scope:
        vv = ThetaPrime(vsize)

        xx = RProjConv2D(offset_creator_class, vv, 96, kernel_size=3, strides=1, kernel_initializer='he_normal',
                         padding='same', activation='relu', kernel_regularizer=l2(weight_decay))(preproc_images)
        xx = RProjConv2D(offset_creator_class, vv, 96, kernel_size=3, strides=1, kernel_initializer='he_normal',
                         padding='same', activation='relu', kernel_regularizer=l2(weight_decay))(xx)
        xx = RProjConv2D(offset_creator_class, vv, 96, kernel_size=3, strides=2, kernel_initializer='he_normal',
                         padding='same', activation='relu', kernel_regularizer=l2(weight_decay))(xx)

        xx = RProjConv2D(offset_creator_class, vv, 192, kernel_size=3, strides=1, kernel_initializer='he_normal',
                         padding='same', activation='relu', kernel_regularizer=l2(weight_decay))(xx)
        xx = RProjConv2D(offset_creator_class, vv, 192, kernel_size=3, strides=1, kernel_initializer='he_normal',
                         padding='same', activation='relu', kernel_regularizer=l2(weight_decay))(xx)
        xx = RProjConv2D(offset_creator_class, vv, 192, kernel_size=3, strides=2, kernel_initializer='he_normal',
                         padding='same', activation='relu', kernel_regularizer=l2(weight_decay))(xx)

        xx = RProjConv2D(offset_creator_class, vv, 192, kernel_size=3, strides=1, kernel_initializer='he_normal',
                         padding='same', activation='relu', kernel_regularizer=l2(weight_decay))(xx)
        xx = RProjConv2D(offset_creator_class, vv, 192, kernel_size=1, strides=1, kernel_initializer='he_normal',
                         padding='valid', activation='relu', kernel_regularizer=l2(weight_decay))(xx)
        xx = RProjConv2D(offset_creator_class, vv, 10, kernel_size=1, strides=1, kernel_initializer='he_normal',
                         padding='valid', activation='relu', kernel_regularizer=l2(weight_decay))(xx)

        xx = GlobalAveragePooling2D()(xx)
        logits = RProjDense(offset_creator_class, vv, 10, kernel_regularizer=l2(weight_decay), activation='softmax')(xx)

        model = ExtendedModel(input=input_images, output=logits)
        model.add_extra_trainable_weight(vv.var_2d)

    nontrackable_fields = ['input_images', 'preproc_images', 'input_labels', 'logits']
    for field in nontrackable_fields:
        model.add_var(field, locals()[field])

    make_and_add_losses(model, input_labels)

    return model

vv = None
def build_test():
    im_shape = (28, 28, 1)
    n_label_vals = 10
    im_dtype = 'float32'

    input_images = Input(shape=im_shape)
    global vv
    if vv is None:
        vv = ThetaPrime(100)

    xx = input_images
    xx = Flatten()(xx)
    for _ in range(3):
        xx = Dense(100, activation='relu')(xx)
    logits = Dense(100)(xx)
    logits = MultiplyLayer(vv.var)(logits)
    logits = Dense(10)(logits)
    model = Model(inputs=input_images, outputs=logits)

    return model, vv

def build_LeNet_cifar(weight_decay=0, vsize=100, shift_in=None, proj_type='sparse'):
    im_shape = (32, 32, 3)
    n_label_vals = 10
    im_dtype = 'float32'

    assert proj_type in ('dense', 'sparse')
    if proj_type == 'dense':
        offset_creator_class = OffsetCreatorDenseProj
    else:
        # sparse
        offset_creator_class = OffsetCreatorSparseProj

    with tf.name_scope('inputs'):
        input_images, preproc_images = make_image_input_preproc(im_shape, im_dtype, shift_in=shift_in)
        input_labels = Input(batch_shape=(None,), dtype='int64')

    with tf.name_scope('net') as scope:
        vv = ThetaPrime(vsize)
        xx = RProjConv2D(offset_creator_class, vv, 6, kernel_size=5, strides=1, kernel_initializer='he_normal', padding='valid', activation='relu', kernel_regularizer=l2(weight_decay))(preproc_images)
        xx = MaxPooling2D((2, 2))(xx)
        xx = RProjConv2D(offset_creator_class, vv, 16, kernel_size=5, strides=1, kernel_initializer='he_normal', padding='valid', activation='relu', kernel_regularizer=l2(weight_decay))(xx)
        xx = MaxPooling2D((2, 2))(xx)
        xx = Flatten()(xx)
        # xx = Dropout(0.5)(xx)
        xx = RProjDense(offset_creator_class, vv, 120, kernel_initializer='he_normal', activation='relu', kernel_regularizer=l2(weight_decay))(xx)
        # xx = Dropout(0.5)(xx)
        xx = RProjDense(offset_creator_class, vv, 84, kernel_initializer='he_normal', activation='relu', kernel_regularizer=l2(weight_decay))(xx)
        # xx = Dropout(0.5)(xx)
        logits = RProjDense(offset_creator_class, vv, 10, kernel_initializer='glorot_uniform', activation='softmax', kernel_regularizer=l2(weight_decay))(xx)
        model = ExtendedModel(input=input_images, output=logits)
        model.add_extra_trainable_weight(vv.var_2d)

    nontrackable_fields = ['input_images', 'preproc_images', 'input_labels', 'logits']
    for field in nontrackable_fields:
        model.add_var(field, locals()[field])

    make_and_add_losses(model, input_labels)
    return model


def build_model_cifar_LeNet_fastfood(weight_decay=0, vsize=100, shift_in=None, DD=None, d_rate=0.0, c1=6, c2=16, d1=120, d2=84):
    '''If DD is not specified, it will be computed.'''

    im_shape = (32, 32, 3)
    n_label_vals = 10
    im_dtype = 'float32'

    with tf.name_scope('inputs'):
        input_images, preproc_images = make_image_input_preproc(im_shape, im_dtype, shift_in=shift_in)
        input_labels = Input(batch_shape=(None,), dtype='int64')

    def define_model(input_images, DenseLayer, Conv2DLayer):
        vv = ThetaPrime(vsize)
        xx = Conv2DLayer(c1, kernel_size=5, strides=1, kernel_initializer='he_normal', padding='valid', activation='relu', kernel_regularizer=l2(weight_decay))(preproc_images)
        xx = MaxPooling2D((2, 2))(xx)
        xx = Conv2DLayer(c2, kernel_size=5, strides=1, kernel_initializer='he_normal', padding='valid', activation='relu', kernel_regularizer=l2(weight_decay))(xx)
        xx = MaxPooling2D((2, 2))(xx)
        xx = Flatten()(xx)
        xx = Dropout(d_rate)(xx)
        xx = DenseLayer(d1, kernel_initializer='he_normal', activation='relu', kernel_regularizer=l2(weight_decay))(xx)
        xx = Dropout(d_rate)(xx)
        xx = DenseLayer(d2, kernel_initializer='he_normal', activation='relu', kernel_regularizer=l2(weight_decay))(xx)
        xx = Dropout(d_rate)(xx)
        logits = DenseLayer(10, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(xx)
        model = ExtendedModel(input=input_images, output=logits)
        nontrackable_fields = ['input_images', 'preproc_images', 'input_labels', 'logits']
        for field in ['logits']:
            model.add_var(field, locals()[field])
        return model

    if not DD:
        with tf.name_scope('net_disposable'):
            # Make disposable direct model
            model_disposable = define_model(input_images, Dense, Conv2D)

            DD = np.sum([np.prod(var.get_shape().as_list()) for var in model_disposable.trainable_weights]).item()
            print(f"D {DD} {type(DD)}")
            del model_disposable


    with tf.name_scope('net'):
        # Make real RProj FWH model
        fwh_projector = FastWalshHadamardProjector(vsize, DD)

        DenseLayer = lambda *args, **kwargs: RProjDense(OffsetCreatorFastfoodProj, fwh_projector, *args, **kwargs)
        Conv2DLayer = lambda *args, **kwargs: RProjConv2D(OffsetCreatorFastfoodProj, fwh_projector, *args, **kwargs)

        model = define_model(input_images, DenseLayer, Conv2DLayer)
        fwh_projector.check_usage()

        for ww in fwh_projector.trainable_weights:
            model.add_extra_trainable_weight(ww)
        for ww in fwh_projector.non_trainable_weights:
            model.add_extra_non_trainable_weight(ww)

    nontrackable_fields = ['input_images', 'preproc_images', 'input_labels']
    for field in nontrackable_fields:
        model.add_var(field, locals()[field])

    make_and_add_losses(model, input_labels)

    return model