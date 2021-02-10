
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import apply_affine_transform


def augment(image,label):
    image = tf.image.random_flip_left_right(image)
    image = tf.numpy_function(shift, [image], tf.float32)
    image = normalize(image)
    # debug(image, label)

    return image, label

def test_augment(image,label):
    return normalize(image), label

def train_aux_augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.numpy_function(shift, [image], tf.float32)
    # image = tf.add(image, tf.random.normal(tf.shape(image), 0, 0.05))
    return image, label

def test_aux_augment(image, label):
    """Augmentation if aux test set is small"""
    return augment(image, label) # same as training

def normalize(image):
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]

    # tf.print("Before:", tf.shape(image), tf.math.reduce_std(image))

    # image = tf.image.per_image_standardization(image)
    # image = image - tf.reshape(mean, [1, 1, 1, 3])
    # image = image / tf.reshape(std, [1, 1, 1, 3])

    # tf.print("After:", tf.shape(image), tf.math.reduce_std(image))

    return image

def shift(images):
    return np.array([shift_single(i) for i in images])

def shift_single(image):
  """ Expects numpy, single image """
  shape = image.shape
  tx = np.random.uniform(-0.1, 0.1) * shape[0]
  ty = np.random.uniform(-0.1, 0.1) * shape[1]

  image = apply_affine_transform(image, 0,
                                   tx, # tx
                                   ty,
                                   0,
                                   1,
                                   1,
                                   row_axis=0,
                                   col_axis=1,
                                   channel_axis=2,
                                   fill_mode='nearest')
  return image

def debug(image, label):
    for i in range(image.shape[0]):
        plt.figure()
        plt.imshow(image[i] + 0.5)
        plt.title(f"Label: {label[i]}")
        plt.show()