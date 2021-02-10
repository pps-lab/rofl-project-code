import tensorflow as tf


class EvasionMethod(object):

    def __init__(self, alpha):
        """

        :type alpha: float|None alpha weight of evasion method. The closer to 1 the more we want to evade.
        """
        self.alpha = alpha

    def loss_term(self, model):
        return None

    def update_after_batch(self, model):
        return

    def update_after_training(self, model):
        return


