from keras import optimizers


def build_optimizer(optimizer, lr):
    if lr is None:
        return optimizer
    if optimizer == 'sgd':
        return optimizers.SGD(lr=lr)
    elif optimizer == 'adam':
        return optimizers.Adam(lr=lr)
    elif optimizer == 'adadelta':
        return optimizers.Adadelta(learning_rate=lr)
    else:
        raise NotImplementedError(f"Optimizer {optimizer} with learning rate {lr} not supported.")