import tensorflow as tf
import numpy as np

from src.data.tf_data_global import IIDGlobalDataset
from src.attack.evasion.norm import NormBoundPGDEvasion
from src.attack.evasion.trimmed_mean import TrimmedMeanEvasion
from src.attack.attack import AttackDatasetBridge
from src.attack.untargeted_attack import UntargetedAttack
from src.attack.targeted_attack import TargetedAttack
from src.data.tf_data import ImageGeneratorDataset, Dataset


class AttackTest(tf.test.TestCase):

    def setUp(self):
        super(AttackTest, self).setUp()

        self.model = tf.keras.models.load_model("./../../../models/lenet5_emnist_098.h5")
        (x_train, y_train), (x_test, y_test) = Dataset.get_emnist_dataset(-1, 1)
        (x_train, y_train), (x_test, y_test) = (x_train[0], y_train[0]), (x_test[0], y_test[0])
        (x_train, y_train) = (x_train[:15000], y_train[:15000])
        targets = [1, 2, 3, 4, 5, 6, 7, 8]
        x_mal, y_mal_orig = x_train[targets], y_train[targets]
        y_mal = np.repeat(3, len(targets)).astype(y_train.dtype)
        np.delete(x_train, targets)
        np.delete(y_train, targets)
        self.global_dataset = IIDGlobalDataset(x_train, y_train, 30, x_test, y_test)
        self.dataset = AttackDatasetBridge(Dataset(x_train, y_train))
        self.dataset.global_dataset.x_aux = x_mal
        self.dataset.global_dataset.y_aux = y_mal_orig
        self.dataset.global_dataset.mal_aux_labels = y_mal

        self.test_accuracy = tf.keras.metrics.Mean(name='test_accuracy')

    def _evaluate_targeted(self):
        batch_x, batch_y = self.dataset.global_dataset.x_aux, self.dataset.global_dataset.mal_aux_labels
        preds = self.model(batch_x, training=False).numpy().argmax(axis=1)
        pred_inds = preds == batch_y

        adv_success = np.mean(pred_inds)
        print(f"Adv success: {adv_success}")

    def _evaluate_untargeted(self):
        for batch_x, batch_y in self.global_dataset.get_test_batch(64, 12):
            self.optimized_evaluate(batch_x, batch_y)

        test_accuracy = self.test_accuracy.result().numpy()
        print(f"Adv success: {1 - test_accuracy}")

    @tf.function
    def optimized_evaluate(self, batch_x, batch_y):
        prediction_tensor = self.model(batch_x, training=False)
        prediction = prediction_tensor
        y_ = tf.cast(tf.argmax(prediction, axis=1), tf.uint8)
        test_accuracy_batch = tf.equal(y_, batch_y)
        self.test_accuracy(tf.reduce_mean(tf.cast(test_accuracy_batch, tf.float32)))

    def tearDown(self):
        pass

    def test_untargeted_attack(self):
        self._evaluate_untargeted()
        att = UntargetedAttack()
        att.set_stealth_method(NormBoundPGDEvasion(self.model.get_weights(), "linf", 0.1, 1, pgd_factor=.1))
        weights = att.generate(self.dataset, self.model,
                               num_epochs=1,
                               optimizer=tf.keras.optimizers.Adam(),
                               loss_object=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
        self.model.set_weights(weights)
        self._evaluate_untargeted()

    def test_untargeted_attack_tootight(self):
        self._evaluate_untargeted()
        att = UntargetedAttack()
        att.set_stealth_method(NormBoundPGDEvasion(self.model.get_weights(), "linf", 0.00001, 1, pgd_factor=0.00001))
        weights = att.generate(self.dataset, self.model,
                               num_epochs=1,
                               optimizer=tf.keras.optimizers.Adam(),
                               loss_object=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                               alpha=0.1)
        self.model.set_weights(weights)
        self._evaluate_untargeted()

    def test_untargeted_attack_trimmedmean(self):
        self._evaluate_untargeted()
        att = UntargetedAttack()
        att.set_stealth_method(TrimmedMeanEvasion(0.5, [self.model.get_weights(), self.model.get_weights(), self.model.get_weights()], 1))
        weights = att.generate(self.dataset, self.model,
                               num_epochs=1,
                               optimizer=tf.keras.optimizers.Adam(),
                               loss_object=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
        self.model.set_weights(weights)
        self._evaluate_untargeted()

    def test_targeted_attack_norm(self):
        self._evaluate_untargeted()
        att = TargetedAttack()
        att.set_stealth_method(NormBoundPGDEvasion(self.model.get_weights(), "linf", 0.1, 1, pgd_factor=.1))
        weights = att.generate(self.dataset, self.model,
                               num_epochs=3,
                               num_batch=6,
                               poison_samples=5,
                               optimizer=tf.keras.optimizers.Adam(),
                               loss_object=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
        self.model.set_weights(weights)
        self._evaluate_targeted()


if __name__ == '__main__':
    tf.test.main()
