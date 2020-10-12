import pickle
import codecs
from keras.models import model_from_json

# Only a subset of the exchanged messages,
# namely those which contain a payload

class Message(object):
    def serialize(self):
        # NOTE mlei: decoding problems if only used with pickle 
        # (think it socketio/engineio's issue... cant find anything in the internet)
        return codecs.encode(pickle.dumps(self), "base64").decode()

def parse_msg(content) -> Message:
    return pickle.loads(codecs.decode(content.encode(), "base64"))


class TransferModelConfigMsg(Message):
    def __init__(self,
                 num_of_clients,
                 client_batch_size,
                 num_local_epochs,
                 optimizer,
                 learning_rate,
                 loss,
                 metrics,
                 image_augmentation,
                 lr_decay,
                 model_id,
                 probabilistic_quantization,
                 fp_bits,
                 fp_frac,
                 range_bits):
        self.num_of_clients = num_of_clients
        self.client_batch_size = client_batch_size
        self.num_local_epochs = num_local_epochs
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.loss = loss
        self.metrics = metrics
        self.image_augmentation = image_augmentation,
        self.lr_decay = lr_decay
        self.model_id = model_id
        self.probabilistic_quantization = probabilistic_quantization
        self.fp_bits = fp_bits
        self.fp_frac = fp_frac
        self.range_bits = range_bits

class TransferModelMsg(Message):
    def __init__(self, model=None, model_name=None, seed=None):
        self.model = model
        self.model_name = model_name
        self.seed = seed

class TransferInitialWeightsMsg(Message):
    def __init__(self, weights):
        self.weights = weights

class TransferCryptoConfigMsg(Message):
    def __init__(self, value_range, n_partition, l2_value_range):
        self.value_range = value_range
        self.n_partition = n_partition
        self.l2_value_range = l2_value_range

class StartTrainingMsg(Message):
    def __init__(self, round_id, weights, content=None):
        self.round_id = round_id
        self.weights = weights
        self.content = content

class TrainingFinishedMsg(Message):
    def __init__(self, model_id, round_id, content=None):
        self.model_id = model_id
        self.round_id = round_id
        self.content = content