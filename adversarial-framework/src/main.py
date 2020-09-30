import tensorflow as tf
import numpy as np

from src.hyperparameter_tuning import tune_hyper
from src.config import get_config
from src.federated_averaging import FederatedAveraging
from src.tf_model import Model


def load_model():
    if args.load_model is not None:
        model = tf.keras.models.load_model(args.load_model) # Load with weights
    else:
        model = Model.create_model(args.model_name, config['intrinsic_dimension'], config['regularization_rate'])
    return model


def main():

    if args.hyperparameter_tuning.lower() == "true":
        tune_hyper(args, config)
    else:
        if not Model.model_supported(args.model_name, args.dataset):
            raise Exception(
                f'Model {args.model_name} does not support {args.dataset}! Check method Model.model_supported for the valid combinations.')

        models = [load_model() for i in range(args.workers)]

        server_model = FederatedAveraging(config, models)
        server_model.init()
        server_model.fit()



if __name__ == '__main__':
    config, args = get_config()
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # tf.profiler.experimental.server.start(6009)

    main()
