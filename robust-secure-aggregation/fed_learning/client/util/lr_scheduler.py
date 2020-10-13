from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
import numpy as np

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def get_callbacks(lr_decay, epoch) -> list:
    if lr_decay == 'cifar_resnet_step':
        lr_scheduler = LearningRateScheduler(specific_decay_cifar(epoch))

        lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                       cooldown=0,
                                       patience=5,
                                       min_lr=0.5e-6,
                                       verbose=1)

        return [lr_scheduler, lr_reducer]
    if lr_decay == 'mnist_subspace_step':
        lr_scheduler = LearningRateScheduler(specific_decay_mnist(epoch))

        return [lr_scheduler]
    else:
        return []

def specific_decay_cifar(epoch):
    def call(e):
        return lr_schedule_cifar(epoch)
    return call

def specific_decay_mnist(epoch):
    def call(e):
        return lr_schedule_cifar(epoch)
    return call

# def get_callbacks(lr_decay, round) -> list:
#     if lr_decay == 'cifar_resnet_step':
#         lr_scheduler = LearningRateScheduler(lr_schedule)
#
#         lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
#                                        cooldown=0,
#                                        patience=5,
#                                        min_lr=0.5e-6,
#                                        verbose=1)
#
#         return [lr_scheduler, lr_reducer]
#     else:
#         return []


def lr_schedule_cifar(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3 # federated
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 60:
        lr *= 1e-3
    elif epoch > 40:
        lr *= 1e-2
    elif epoch > 20:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

def lr_schedule_mnist(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 0.01 # federated
    if epoch > 40:
        lr *= 1e-1
    logger.info(f'Learning rate: {lr} {epoch}')
    return lr