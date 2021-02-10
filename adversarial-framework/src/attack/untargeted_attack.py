from src.attack.attack import LossBasedAttack

import tensorflow as tf
import logging

logger = logging.getLogger(__name__)

class UntargetedAttack(LossBasedAttack):

    def generate(self, dataset, model, **kwargs):

        self.parse_params(**kwargs)

        self.weights = model.get_weights()

        loss_object_with_reg = self._combine_losses(
            self.stealth_method.loss_term(model) if self.stealth_method is not None else None,
            self.stealth_method.alpha)

        for epoch in range(self.num_epochs):
            logger.info(f"Epoch {epoch}")
            for batch_x, batch_y in dataset.get_data():
                with tf.GradientTape() as tape:
                    loss_value = loss_object_with_reg(y_true=batch_y, y_pred=model(batch_x, training=True))
                    grads = self._compute_gradients(tape, loss_value, model)

                    for k in range(len(grads)):
                        grads[k] = -grads[k]
                    self.optimizer.apply_gradients(zip(grads, model.trainable_variables))

                if self.step_decay is not None:
                    self.step_decay.apply_step()

                if self.stealth_method is not None:
                    self.stealth_method.update_after_batch(model)

        if self.stealth_method is not None:
            self.stealth_method.update_after_training(model)

        return model.get_weights()

    def parse_params(self, num_epochs, optimizer, loss_object, step_decay=None):
        self.num_epochs = num_epochs
        self.optimizer = optimizer
        self.loss_object = loss_object
        self.step_decay = step_decay
