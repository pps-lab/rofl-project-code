from keras_preprocessing.image import ImageDataGenerator


def get_augmentation(image_augmentation):
    if image_augmentation == 'cifar':
        return ImageDataGenerator(
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True
        )
    else:
        return None

