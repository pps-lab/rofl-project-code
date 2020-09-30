from ..util.optimizer import build_optimizer


class LocalModelConfig(object):
    
    def __init__(self,
                 num_of_clients,
                 client_batch_size,
                 num_local_epochs,
                 optimizer,
                 lr,
                 loss,
                 metrics,
                 model_id,
                 probabilistic_quantization,
                 fp_bits,
                 fp_frac,
                 range_bits):
        self.num_of_clients = num_of_clients
        self.client_batch_size = client_batch_size
        self.num_local_epochs = num_local_epochs
        self.optimizer = optimizer
        self.learning_rate = lr
        self.loss = loss
        self.metrics = metrics
        self.model_id = model_id
        self.probabilistic_quantization = probabilistic_quantization
        self.fp_bits = fp_bits
        self.fp_frac = fp_frac
        self.range_bits = range_bits

    def get_optimizer(self):
        return build_optimizer(self.optimizer, self.learning_rate)

    @classmethod
    def from_config(cls, config, model_id):
        return cls(config.num_clients,
                   config.client_batch_size,
                   config.num_local_epochs,
                   config.optimizer,
                   config.learning_rate,
                   config.loss,
                   config.metrics,
                   model_id)


