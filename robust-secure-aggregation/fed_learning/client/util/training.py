from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from keras_preprocessing.image import ImageDataGenerator

from fed_learning.client.dataset import DataSet
from fed_learning.client.local_model_config import LocalModelConfig

import numpy as np

from fed_learning.client.util import lr_scheduler, augmentation


class LocalModelTraining:

    def __init__(self, dataset: DataSet, model_config: LocalModelConfig):
        self.dataset = dataset
        self.model_config = model_config
        self.model = None

    def set_model(self, model):
        self.model = model

    def train(self):
        """Trains model based on configuration"""
        callbacks = lr_scheduler.get_callbacks(self.model_config.lr_decay)
        augmentation_generator = augmentation.get_augmentation(self.model_config.image_augmentation)
        verbose = 1
        print(self.dataset.x_train[0])
        print("Test", self.dataset.x_test[0])
        if augmentation_generator is not None:
            # print("Augmenting")
            self.model.fit_generator(augmentation_generator.flow(self.dataset.x_train, self.dataset.y_train,
                                                                 batch_size=self.model_config.client_batch_size),
                                     epochs=self.model_config.num_local_epochs,
                                     validation_data=(self.dataset.x_test, self.dataset.y_test),
                                     verbose=verbose,
                                     callbacks=callbacks)
        else:
            # print("Regular fix")
            self.model.fit(self.dataset.x_train,
                           self.dataset.y_train,
                           batch_size=self.model_config.client_batch_size,
                           epochs=self.model_config.num_local_epochs,
                           validation_data=(self.dataset.x_test, self.dataset.y_test),
                           verbose=verbose,
                           callbacks=callbacks)
