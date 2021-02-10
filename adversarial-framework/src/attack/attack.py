
from src.data.tf_data import Dataset
from src.attack.evasion.evasion_method import EvasionMethod

class Attack(object):

    def __init__(self):
        pass

    def generate(self, dataset, model, **kwargs):
        raise NotImplementedError("Sub-classes must implement generate.")
        return x

    def _compute_gradients(self, tape, loss_value, model):
        grads = tape.gradient(loss_value, model.trainable_variables)
        return grads

class StealthAttack(Attack):

    def __init__(self):
        super().__init__()
        self.stealth_method = None


    def set_stealth_method(self, stealth_method: EvasionMethod):
        """

        :type stealth_method: EvasionMethod|None
        """
        self.stealth_method = stealth_method

class LossBasedAttack(StealthAttack):

    def _combine_losses(self, reg, alpha):
        """
        Combine loss with regularization loss.
        :param reg: callable regularization loss callback function
        :param alpha: float|None
        :return:
        """
        if reg is None or alpha is None:
            def direct_loss(y_true, y_pred):
                return self.loss_object(y_true=y_true, y_pred=y_pred)
            return direct_loss

        def loss(y_true, y_pred):
            return alpha * self.loss_object(y_true=y_true, y_pred=y_pred) + \
                   ((1 - alpha) * reg(y_true=y_true, y_pred=y_pred))
        return loss


class AttackDataset(object):

    def get_data(self):
        raise NotImplementedError()

class AttackDatasetBridge(AttackDataset):

    def __init__(self, global_dataset: Dataset):
        self.global_dataset = global_dataset

    def get_data_with_aux(self, poison_samples, num_batch):
        return self.global_dataset.get_data_with_aux(poison_samples, num_batch)

    def get_data(self):
        return self.global_dataset.get_data()


