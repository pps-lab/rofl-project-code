import os

from tensorboard.plugins.hparams import api as hp
import tensorflow as tf
import numpy as np

from src.federated_averaging import FederatedAveraging
from src.tf_model import Model

def load_model(args, config):
    if args.load_model is not None:
        model = tf.keras.models.load_model(args.load_model) # Load with weights
    else:
        model = Model.create_model(args.model_name, config['intrinsic_dimension'], config['regularization_rate'])
    return model

def tune_hyper(args, config):

    HP_MAL_NUM_BATCH = hp.HParam('mal_num_batch', hp.Discrete(args.mal_num_batch))

    mal_lr = args.mal_learning_rate if isinstance(args.mal_learning_rate, list) and len(args.mal_learning_rate) > 0 else [args.learning_rate]
    HP_MAL_LR = hp.HParam('mal_learning_rate', hp.Discrete(mal_lr))
    HP_WEIGHT_REG = hp.HParam('weight_regularization_alpha', hp.Discrete(args.weight_regularization_alpha))

    HP_WEIGHT_SCALE = hp.HParam('scale_attack_weight', hp.Discrete(args.scale_attack_weight))

    # NUM_ClIENTS = hp.HParam('mal_learning_rate', hp.Discrete(args.mal_learning_rate))
    HP_NUM_CLIENTS_SETUP = hp.HParam('num_clients_attack', hp.Discrete(args.tune_attack_clients))

    METRIC_ACCURACY = 'evaluation/test_accuracy'
    METRIC_ADV_SUCCESS = 'evaluation/adv_success'

    experiment_root_dir = os.path.join(os.getcwd(), 'experiments')
    experiment_dir = os.path.join(experiment_root_dir, args.experiment_name)

    with tf.summary.create_file_writer(experiment_dir).as_default():
        hp.hparams_config(
            hparams=[HP_MAL_NUM_BATCH, HP_MAL_LR, HP_WEIGHT_REG, HP_WEIGHT_SCALE, HP_NUM_CLIENTS_SETUP],
            metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy'),
                     hp.Metric(METRIC_ADV_SUCCESS, display_name='Adversarial Success')],
        )

    session_num = 0
    for mal_lr in HP_MAL_LR.domain.values:
        for mal_num_batch in HP_MAL_NUM_BATCH.domain.values:
            for wr in HP_WEIGHT_REG.domain.values:
                for scale in HP_WEIGHT_SCALE.domain.values:
                    for num_clients_att in HP_NUM_CLIENTS_SETUP.domain.values:
                        hparams_dict = {
                            HP_MAL_NUM_BATCH.name: mal_num_batch,
                            HP_MAL_LR.name: mal_lr,
                            HP_WEIGHT_REG.name: wr,
                            HP_WEIGHT_SCALE.name: scale,
                            HP_NUM_CLIENTS_SETUP.name: num_clients_att
                        }
                        config_run = config
                        config_run["mal_num_batch"] = mal_num_batch
                        config_run["mal_learning_rate"] = mal_lr
                        config_run["weight_regularization_alpha"] = wr
                        if num_clients_att != -1:
                            # glob_lr = args.global_learning_rate if args.global_learning_rate == -1
                            selected = int(num_clients_att * args.tune_attack_clients_selected_frac)
                            config_run["num_selected_clients"] = selected
                            config_run["num_clients"] = num_clients_att
                            config_run["scale_attack_weight"] = num_clients_att / args.global_learning_rate # assumes nom. learning_rate
                            # TODO: Autocalc global lr for full scale
                            # if args.global_learning_rate == -1:
                            #     config_run["scale_attack_weight"] = num_clients_att / selected
                            # else:
                            #     config_run["scale_attack_weight"] = num_clients_att / selected
                        else:
                            config_run["scale_attack_weight"] = scale

                        run = f"run-{session_num}"
                        run_dir = os.path.join(experiment_dir, run)
                        run_dir = os.path.join(run_dir, "events")
                        with tf.summary.create_file_writer(run_dir).as_default():
                            hp.hparams(hparams_dict)  # record the values used in this trial

                        print(hparams_dict)

                        np.random.seed(args.seed)
                        tf.random.set_seed(args.seed)

                        if not Model.model_supported(args.model_name, args.dataset):
                            raise Exception(
                                f'Model {args.model_name} does not support {args.dataset}! Check method Model.model_supported for the valid combinations.')

                        models = [load_model(args, config) for i in range(args.workers)]

                        server_model = FederatedAveraging(config, models, run)
                        server_model.init()
                        server_model.fit()

                        accuracy, adv_success = server_model.evaluate()

                        # with tf.summary.create_file_writer(run_dir).as_default():
                        #     tf.summary.scalar(METRIC_ACCURACY, accuracy, server_model.num_rounds)
                        #     tf.summary.scalar(METRIC_ADV_SUCCESS, adv_success, server_model.num_rounds)

                        session_num += 1

                        metrics_dict = {
                            METRIC_ACCURACY: accuracy,
                            METRIC_ADV_SUCCESS: adv_success
                        }
                        server_model.write_hparams(hparams_dict, metrics_dict)