import numpy as np
import tensorflow as tf


class StepDecay:
    def __init__(self, initial_lr, total_steps):
        # store the base initial learning rate, drop factor, and
        # epochs to drop every
        self.initial_lr = initial_lr
        self.total_steps = total_steps
        self.step = 0

    def __call__(self):
        lr = self.initial_lr
        if self.step > 0.8 * self.total_steps:
            lr = lr * 0.01
        elif self.step > 0.2 * self.total_steps:
            lr = lr * 0.1

        self.step += 1

        # print(f"LR: {lr}")

        return float(lr)
