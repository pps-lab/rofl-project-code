
import tensorflow as tf
import matplotlib.pyplot as plt

def augment(image,label):
    image = tf.image.random_flip_left_right(image)

    shape = tf.shape(image)

    # width = int(shape[1])
    # height = int(shape[2])
    width = 32
    height = 32
    image = tf.image.resize_with_crop_or_pad(image, 38, 38) # Add 6 pixels of padding
    image = tf.image.random_crop(image, [shape[0], width, height, shape[3]])

    # debug(image, label)

    return image, label

def debug(image, label):
    plt.figure()
    plt.imshow(image[0])
    plt.title(f"Label: {label[0]}")
    plt.show()