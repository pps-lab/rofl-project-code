import numpy as np
import tensorflow as tf


class StepDecay:
    def __init__(self, initial_lr, total_steps, bound1=0.2, bound2=0.8):
        # store the base initial learning rate, drop factor, and
        # epochs to drop every
        self.initial_lr = initial_lr
        self.total_steps = total_steps
        self.step = 0
        self.mul = 1
        self.bound1 = bound1
        self.bound2 = bound2

    def __call__(self):
        lr = self.initial_lr * self.mul
        if self.step > self.bound2 * self.total_steps:
            lr = lr * 0.01
        elif self.step > self.bound1 * self.total_steps:
            lr = lr * 0.1

        # print(f"LR: {lr}")

        return float(lr)

    def apply_step(self):
        self.step += 1
