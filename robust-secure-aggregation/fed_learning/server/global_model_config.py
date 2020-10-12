import uuid

from ..util.optimizer import build_optimizer


class GlobalModelConfig(object):

    def __init__(self,
                 iterations,
                 num_clients,
                 client_batch_size,
                 num_local_epochs,
                 optimizer,
                 lr,
                 loss,
                 metrics: list,
                 image_augmentation,
                 lr_decay,
                 probabilistic_quantization,
                 fp_bits,
                 fp_frac,
                 range_bits):
        self.iterations = iterations
        self.num_clients = num_clients
        self.client_batch_size = client_batch_size
        self.num_local_epochs = num_local_epochs
        self.optimizer = optimizer
        self.learning_rate = lr
        self.loss = loss
        self.metrics = metrics
        self.image_augmentation = image_augmentation
        self.lr_decay = lr_decay
        self.model_id = str(uuid.uuid4())
        self.probabilistic_quantization = probabilistic_quantization
        self.fp_bits = fp_bits
        self.fp_frac = fp_frac
        self.range_bits = range_bits

    @classmethod
    def from_config(cls, config):
        return cls(config.iterations,
                   config.num_clients,
                   config.client_batch_size,
                   config.num_local_epochs,
                   config.optimizer,
                   config.learning_rate,
                   config.loss,
                   config.metrics,
                   config.image_augmentation,
                   config.lr_decay,
                   config.model_id, # ??
                   config.probabilistic_quantization,
                   config.fp_bits,
                   config.fp_frac,
                   config.range_bits)