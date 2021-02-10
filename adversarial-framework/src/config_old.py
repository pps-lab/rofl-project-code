import sys

import configargparse
import logging

from src.client_attacks import Attack

parser = configargparse.ArgumentParser()
parser.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

# logging configuration
parser.add_argument(
    '-d', '--debug',
    help="Print debug statements",
    action="store_const", dest="loglevel", const=logging.DEBUG,
    default=logging.WARNING,
)
parser.add_argument(
    '-v', '--verbose',
    help="Print verbose",
    action="store_const", dest="loglevel", const=logging.INFO,
)

# client configuration
parser.add_argument('--num_clients', type=int, default=3, help='Total number of clients.')
parser.add_argument('--num_selected_clients', type=int, default=-1,
                    help='The number of selected clients per round; -1 to use all clients.')
parser.add_argument('--num_malicious_clients', type=int, default=0, help='Total number of malicious clients.')
parser.add_argument('--augment_data', type=str, default='false',
                    help='Whether to augment train/aux datasets',
                    choices=['true', 'false'])

# attacks
parser.add_argument('--attacks_config', type=str, default=None, help='Path to attack config.')

parser.add_argument('--attack_type', type=str, default='untargeted', help='Attack type.',
                    choices=['untargeted', 'backdoor', Attack.DEVIATE_MAX_NORM.value])
parser.add_argument('--estimate_other_updates', type=str, default='false',
                    help='Whether to estimate the update of the others.',
                    choices=['true', 'false'])
parser.add_argument('--attack_after', type=int, default=0, help='After which round to start behaving maliciously.')
parser.add_argument('--attack_stop_after', type=int, default=10000000, help='After which round to stop behaving maliciously.')
parser.add_argument('--attack_frequency', type=float, default=None, help='Frequency of malicious parties being selected. Default is None, for random selection')
parser.add_argument('--weight_regularization_alpha', type=float, default=[1], nargs='+',
                    help='Alpha value for weight regularization. Keep one for none.')
parser.add_argument('--attacker_full_dataset', type=str, default='false',
                    help='Whether the attack can access the full dataset',
                    choices=['true', 'false'])
parser.add_argument('--attacker_full_knowledge', type=str, default='false',
                    help='Whether the attacker has access to the benign updates in a specific round',
                    choices=['true', 'false'])
parser.add_argument('--permute_dataset', type=int, nargs='+', default=[], help='Use with caution. Run many attacks while permuting items in this list')

# attacks - untargeted
parser.add_argument('--untargeted_after_training', type=str, default='false',
                    help='Whether local model gradients are flipped in each local training iteration or when the local model is fully trained.',
                    choices=['true', 'false'])

# attacks - targeted_deterministic_attack
parser.add_argument('--targeted_deterministic_attack_objective', type=int, default=3,
                    help="All malicious clients try to make the model misclassify a given input as this predefined objective. Only Applicable if num_malicious_clients is non-zero value and 'attack_type' is 'targeted_deterministic'.")

# attacks - targeted
parser.add_argument('--targeted_attack_objective', type=int, default=[5, 7], nargs='+',
                    help="Malicious clients try to make the model classify every sample of a class (first arguments) as a target (second argument). Only applicable if num_malicious_clients is non-zero value and 'attack_type' is 'targeted'.")
parser.add_argument('--targeted_attack_benign_first', type=str, default='false', choices=['true', 'false'],
                    help="If set to true, the attack would perform benign training first and fine tune updates on malicious dataset. Applicable if attack_type is 'targeted'.")

# attacks - min loss
parser.add_argument('--aux_samples', type=int, default=-1,
                    help="Size of auxiliary dataset that is used for backdoor attack.")
parser.add_argument('--gaussian_noise', type=float, default=0,
                    help="Sigma value for gaussian noise that is added to aux samples if the value is > 0.")
parser.add_argument('--backdoor_type', type=str, default='semantic', help='Backdoor type. Semantic = backdoor_feature_*, tasks = Sun et al., edge = edge cases',
                    choices=['semantic', 'tasks', 'edge'])
parser.add_argument('--backdoor_stealth', type=str, default='false', help='Whether to use stealth in backdoor.',
                    choices=['true', 'false'])
parser.add_argument('--backdoor_attack_objective', type=int, default=[7, 1], nargs='+',
                    help="What class to mispredict `aux_samples` times. Only applicable if num_malicious_clients is non-zero value and 'attack_type' is 'segment_poisoning'.")
parser.add_argument('--backdoor_tasks', type=int, default=1,
                    help="Number of backdoor tasks to fill")
parser.add_argument('--mal_num_epochs_max', type=int, default=100, help="Maximum number of epochs to run the attack")
parser.add_argument('--mal_target_loss', type=float, default=0.1, help="Target threshold for training")

# attacks - edge case
parser.add_argument('--edge_case_type', type=str, default=None, help='Which edge case class to use')

# attacks - data poisoning
parser.add_argument('--poison_samples', type=int, default=1,
                    help="How many samples to poison in a batch")
parser.add_argument('--mal_num_batch', type=int, default=[200], nargs='+',
                    help="How many batches to run")

# attack - backdoor feature
parser.add_argument('--backdoor_feature_aux_train', type=int, default=[], nargs='+',
                    help="What samples to use as aux train set. Only applicable 'attack_type' is 'segment_poisoning' or 'model_replacement'.")
parser.add_argument('--backdoor_feature_aux_test', type=int, default=[], nargs='+',
                    help="What samples to use as aux test set. Only applicable 'attack_type' is 'segment_poisoning' or 'model_replacement'.")
parser.add_argument('--backdoor_feature_target', type=int, default=2,
                    help="Malicious target label")
parser.add_argument('--backdoor_feature_benign_regular', type=int, default=[], nargs='+',
                    help="Include specific benign samples in training from the dataset")
parser.add_argument('--backdoor_feature_remove_malicious', type=str, default='false', help='Whether to remove the malicious samples from the honest clients.',
                    choices=['true', 'false'])
parser.add_argument('--backdoor_feature_augment_times', type=int, default=0, help="How many times the eval samples should be augmented. Leave 0 for no augmentation")

# attack - backdoor contamination model
parser.add_argument('--contamination_model', action='store_true', default=False,
                    help='Whether attackers modify only a subset of neurons')
parser.add_argument('--contamination_rate', type=float, default=[None], nargs='+',
                    help='Percentage of neurons (filters) per layer that is modified by adversaries.'
                         'If only one value is specified, then the same contamination rate is used for all '
                         'convolutional and dense layers.')

# attacks - PGD
parser.add_argument('--pgd', type=str, default=None, choices=['l2', 'l_inf'],
                    help='(Projected Gradient Descent)'
                         'Weather malicious clients project their gradients onto the feasible set. '
                         'Compatible with all implemented attacks.')
parser.add_argument('--pgd_constraint', type=float, default=None,
                    help='Projection bound (applicable only if `pgd` is set).')
parser.add_argument('--pgd_clip_frequency', type=int, default=1,
                    help='Clip every x steps of SGD. Defaults to 1 (after every step).')
parser.add_argument('--pgd_adaptive', type=str, default="false", help="Whether to be adaptive in the gradient clipping (not sure if working).")


# attacks - boosting supplement
parser.add_argument('--scale_attack', type=str, default="false", help="Whether malicious clients scale their updates.")
parser.add_argument('--scale_attack_weight', type=float, default=[1.0], nargs='+',
                    help="A scaling factor for malicious clients' updates. Only applicable if scale_attack is set to true.")

# defense
parser.add_argument("--clip", type=float, default=None, help="A positive float value for absolute update clipping.")
parser.add_argument("--clip_l2", type=float, default=None, help="A positive float value for l2 update clipping.")
parser.add_argument("--clip_probability", type=float, default=1.0, help="Percentage of weights to clip")
parser.add_argument("--clip_layers", type=int, default=[], nargs='+', help="Indexes of layers to clip. Leave empty for all")

# data configuration
parser.add_argument("--data_distribution", type=str, default='IID', help="IID or non-IID.")
parser.add_argument("--number_of_samples", type=int, default=-1,
                    help="How many samples to use for training; default value of -1 indicates to use the full dataset.")
parser.add_argument("--dataset", type=str, default='mnist', help="Which dataset to use.", choices=['mnist', 'femnist', 'fmnist', 'cifar10'])

# training configuration
parser.add_argument("--model_name", type=str, default='dev', help="Which model to use.",
                    choices=['dev', 'mnist_cnn', 'bhagoji', 'resnet18', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet18_v2', 'resnet56_v2', 'dev_intrinsic', 'dev_fc_intrinsic', 'bhagoji_intrinsic', 'mnistcnn_intrinsic', 'lenet5_mnist', 'lenet5_cifar', 'lenet5_intrinsic', 'allcnn', 'allcnn_intrinsic'])
parser.add_argument("--num_rounds", type=int, default=40, help="Number of training rounds.")
parser.add_argument("--num_epochs", type=int, default=3, help="Number of client epochs.")
parser.add_argument("--num_test_batches", type=int, default=-1, help="Number of test batches to evaluate. -1 for max.")
parser.add_argument("--batch_size", type=int, default=128, help="Clients' batch size.")
parser.add_argument('--optimizer', type=str, default='Adam', help='Which optimizer to use.', choices=['Adam', 'SGD'])
parser.add_argument('--learning_rate', type=float, default=0.0001, nargs="+", help='Learning rate for selected optimizer.')
parser.add_argument('--lr_decay', type=str, default='None', help='Apply decay to the learning rate.',
                    choices=['None', 'exponential', 'boundaries'])
parser.add_argument('--decay_steps', type=float, default=None, help='Decay steps for exponential decay.')
parser.add_argument('--decay_rate', type=float, default=None, help='Decay rate for exponential decay.')
parser.add_argument('--decay_boundaries', type=int, default=[], nargs="+", help='Boundaries for boundaries decay mode')
parser.add_argument('--decay_values', type=float, default=[], nargs="+", help='Values for boundaries decay mode')

parser.add_argument('--regularization_rate', type=float, default=None, help='Weight regularization rate.')

parser.add_argument('--mal_learning_rate', type=float, default=[], nargs="+", help='Malicious learning rate for selected optimizer.')
parser.add_argument('--mal_decay_steps', type=float, default=None, help='Malicious decay steps for exponential decay.')
parser.add_argument('--mal_decay_rate', type=float, default=None, help='Malicious decay rate for exponential decay.')
parser.add_argument('--mal_num_epochs', type=int, default=None, help='How many malicious epochs to run')
parser.add_argument('--mal_step_learning_rate', type=str, default='false', help='Whether to step the learning rate.',
                    choices=['true', 'false'])

parser.add_argument('--federated_dropout_rate', type=float, default=1.0,
                    help='Percentage of neurons (or filters for convolutional layers) that are kept on each layer.')
parser.add_argument('--federated_dropout_all_parameters', action='store_true', default=False,
                    help='If set to True, applies dropout on all parameters randomly according to the dropout rate.'
                         'Applicable only if federated_dropout_rate < 1.0.')
parser.add_argument('--federated_dropout_nonoverlap', action='store_true', default=False,
                    help="Each client receives a unique mask that is not overlapped with other clients' masks."
                         'Applicable only if federated_dropout_rate < 1.0.')
parser.add_argument('--federated_dropout_randommask', type=str, default='false',
                    help="Enable low rank mode instead of federated dropout, i.e. only mask the uplink.")

parser.add_argument('--global_gaussian_noise', type=float, default=0.0,
                    help='Gaussian noise to add to the global model for the server.')
parser.add_argument('--global_learning_rate', type=float, default=-1, help='Global learning rate for the server.')

parser.add_argument("--aggregator", type=str, default='FedAvg', help="Aggregator type. Supported: FedAvg, TrimmedMean")
parser.add_argument('--trimmed_mean_beta', type=float, default=0.1, help='Beta value of trimmed mean. 0 < beta < 1/2.')

parser.add_argument("--intrinsic_dimension", type=int, default=1000, help="Size of intrinsic dimension. Only applicable if using subspace machine learning model.")

parser.add_argument("--load_model", type=str, default=None, help="Path to load an existing model to initialize the setup.")
parser.add_argument('--ignore_malicious_update', type=str, default="false", help="Whether to ignore malicious updates in training.")

parser.add_argument('--quantization', type=str, default=None, help='Whether to use (probabilistic) quantization', choices=['deterministic', 'probabilistic', 'd', 'p'])
parser.add_argument('--q_bits', type=int, default=None, help='Number of bits of the fixed-point number to represent the weights for quantization')
parser.add_argument('--q_frac', type=int, default=None, help='Number of fractional bits of the fixed-point number for quantization')



# logging
parser.add_argument("--experiment_name", type=str, default='tmp', help="Sub-directory where the log files are stored.")
parser.add_argument("--print_every", type=int, default=1,
                    help="After how many rounds to perform and log evaluation on test set.")
parser.add_argument("--save_updates", type=str, default='true', help="Whether to save the weight updates. Disable for large models / large number of clients.",
                    choices=['true', 'false'])
parser.add_argument("--save_norms", type=str, default='false', help="Whether to save the norms for all clients",
                    choices=['true', 'false'])
parser.add_argument("--save_weight_distributions", type=str, default='false', help="Whether to save the weight distributions for all clients",
                    choices=['true', 'false'])
parser.add_argument("--keep_history", action='store_true', default=False,
                    help='Whether Server keeps parameter history.'
                         'Warning: It slows down the training because of principle eigenvalue computation.')
parser.add_argument("--save_model_at", type=int, default=[], nargs='+', help="At what rounds to save model.")

# hyperparameter tuning
parser.add_argument("--hyperparameter_tuning", type=str, default='false', help="Whether to use hyperparameter tuning", choices=['true', 'false'])
parser.add_argument("--tune_attack_clients", type=int, nargs='+', default=[-1], help="Helper for hyperparameter tuning to set the number of clients + scale_attack_weight")
parser.add_argument("--tune_attack_clients_selected_frac", type=float, default=None, help="Fraction of clients to be selected")

parser.add_argument("--hyperparameters_tuned", type=str, nargs='+', default=[], help="Which hyperparams are being tuned at the moment")


# experiment reproducibility
parser.add_argument("--seed", type=int, default=0,
                    help="Seed for random functions. Ensures experiment reproducibility.")

# computational optimization
parser.add_argument("--workers", type=int, default=1, help="How many threads to use for client training simulation.")

parser.add_argument("--optimized_training", type=str, default='true', help="Use optimized training loop where possible.", choices=['true', 'false'])


def get_config():
    args = parser.parse_args()

    logging.basicConfig(level=args.loglevel)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # handler = logging.StreamHandler(sys.stdout)
    # handler.setLevel(logging.DEBUG)
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # handler.setFormatter(formatter)
    # root.addHandler(handler)

    config = dict()

    config['num_clients'] = args.num_clients
    if args.num_selected_clients == -1:
        config['num_selected_clients'] = args.num_clients
    else:
        config['num_selected_clients'] = args.num_selected_clients
    config['num_malicious_clients'] = args.num_malicious_clients
    config['augment_data'] = True if args.augment_data.lower() == "true" else False
    config['weight_regularization_alpha'] = args.weight_regularization_alpha[0]

    config['attack_type'] = args.attack_type

    config['untargeted_after_training'] = True if args.untargeted_after_training.lower() == "true" else False
    config['targeted_deterministic_attack_objective'] = args.targeted_deterministic_attack_objective
    config['targeted_attack_objective'] = tuple(args.targeted_attack_objective)
    config['targeted_attack_benign_first'] = True if args.targeted_attack_benign_first.lower() == 'true' else False

    config['scale_attack'] = True if args.scale_attack.lower() == "true" else False
    config['scale_attack_weight'] = args.scale_attack_weight[0]
    config['data_distribution'] = args.data_distribution
    config['estimate_other_updates'] = True if args.estimate_other_updates.lower() == "true" else False

    config['num_rounds'] = args.num_rounds
    config['num_epochs'] = args.num_epochs
    config['mal_num_epochs'] = args.mal_num_epochs if args.mal_num_epochs is not None else args.num_epochs
    config['batch_size'] = args.batch_size
    config['num_test_batches'] = args.num_test_batches if args.num_test_batches > -1 else sys.maxsize

    config['optimizer'] = args.optimizer
    config['learning_rate'] = args.learning_rate[0] if isinstance(args.learning_rate, list) and len(args.learning_rate) > 0 else args.learning_rate
    config['lr_decay'] = args.lr_decay if args.lr_decay != 'None' else None
    config['decay_steps'] = args.decay_steps
    config['decay_rate'] = args.decay_rate
    config['decay_boundaries'] = args.decay_boundaries
    config['decay_values'] = args.decay_values
    config['regularization_rate'] = args.regularization_rate
    config['mal_learning_rate'] = args.mal_learning_rate[0] if len(args.mal_learning_rate) > 0 else config['learning_rate']
    config['mal_decay_steps'] = args.mal_decay_steps if args.mal_decay_steps is not None else args.decay_steps
    config['mal_decay_rate'] = args.mal_decay_rate if args.mal_decay_rate is not None else args.decay_rate
    config['mal_step_learning_rate'] = True if args.mal_step_learning_rate.lower() == "true" else False
    config['aggregator'] = args.aggregator
    config['trimmed_mean_beta'] = args.trimmed_mean_beta
    config['global_learning_rate'] = args.global_learning_rate
    config['global_gaussian_noise'] = args.global_gaussian_noise
    config['federated_dropout_rate'] = args.rate
    config['federated_dropout_all_parameters'] = args.all_parameters
    config['federated_dropout_nonoverlap'] = args.nonoverlap
    config['federated_dropout_randommask'] = True if args.randommask.lower() == "true" else False
    config['intrinsic_dimension'] = args.intrinsic_dimension
    config['ignore_malicious_update'] = True if args.ignore_malicious_update.lower() == "true" else False
    config['quantization'] = args.quantization
    if config['quantization'] == 'p':
        config['quantization'] = 'probabilistic'
    elif config['quantization'] == 'd':
        config['quantization'] = 'deterministic'
    config['q_bits'] = args.q_bits
    config['q_frac'] = args.q_frac
    assert 0 < args.rate <= 1, 'Federated dropout rate must be in (0, 1] range.'

    config['experiment_name'] = args.experiment_name
    config['print_every'] = args.print_every
    config['save_updates'] = True if args.save_updates.lower() == 'true' else False
    config['keep_history'] = args.keep_history
    config['save_model_at'] = args.save_model_at
    config['load_model'] = args.load_model
    config['save_norms'] = True if args.save_norms.lower() == 'true' else False
    config['save_weight_distributions'] = True if args.save_weight_distributions.lower() == 'true' else False

    config['model_name'] = args.model_name

    if args.clip is not None and args.clip != 0:
        assert args.clip > 0, '`clip` parameter must be a non-negative float.'
    config['clip'] = args.clip if args.clip is not None and args.clip != 0 else None
    config['clip_probability'] = args.clip_probability
    config['clip_l2'] = args.clip_l2
    config['clip_layers'] = args.clip_layers

    config['dataset'] = args.dataset
    config['workers'] = args.workers
    config['number_of_samples'] = args.number_of_samples
    config['aux_samples'] = args.aux_samples if args.aux_samples != -1 else sys.maxsize
    config['mal_num_epochs_max'] = args.mal_num_epochs_max
    config['mal_target_loss'] = args.mal_target_loss
    config['backdoor_type'] = args.backdoor_type
    config['backdoor_stealth'] = True if args.backdoor_stealth.lower() == 'true' else False
    config['backdoor_attack_objective'] = None if args.backdoor_attack_objective[0] == -1 else tuple(args.backdoor_attack_objective)
    config['edge_case_type'] = args.edge_case_type
    config['attack_after'] = args.attack_after
    config['attack_stop_after'] = args.attack_stop_after
    config['attack_frequency'] = args.attack_frequency if args.attack_frequency != -1 else None
    config['attacker_full_dataset'] = True if args.attacker_full_dataset.lower() == "true" else False
    config['attacker_full_knowledge'] = True if args.attacker_full_knowledge.lower() == "true" else False
    config['backdoor_tasks'] = args.backdoor_tasks if args.num_malicious_clients > 0 else 0
    config['backdoor_feature_aux_train'] = args.backdoor_feature_aux_train
    config['backdoor_feature_aux_test'] = args.backdoor_feature_aux_test
    config['backdoor_feature_target'] = args.backdoor_feature_target
    config['backdoor_feature_benign_regular'] = args.backdoor_feature_benign_regular
    config['backdoor_feature_remove_malicious'] = True if args.backdoor_feature_remove_malicious.lower() == "true" else False
    config['backdoor_feature_augment_times'] = args.backdoor_feature_augment_times
    config['poison_samples'] = args.poison_samples
    config['mal_num_batch'] = args.mal_num_batch[0]
    config['optimized_training'] = True if args.optimized_training.lower() == "true" else False

    assert args.gaussian_noise >= 0.
    config['gaussian_noise'] = args.gaussian_noise

    config['contamination_model'] = args.contamination_model
    config['contamination_rate'] = _preprocess_contamination_rate(args)
    if args.pgd is not None:
        assert args.pgd_constraint is not None, "PGD constraint value must be set."
    config['pgd'] = args.pgd
    config['pgd_constraint'] = args.pgd_constraint
    config['pgd_clip_frequency'] = args.pgd_clip_frequency
    config['pgd_adaptive'] = True if args.pgd_adaptive.lower() == 'true' else False
    logging.info(config)

    logging.warning("Can I see this?")

    return config, args


def _preprocess_contamination_rate(args):
    if not args.contamination_model:
        return args.contamination_rate

    assert args.contamination_rate[0] is not None, "Contamination rate must be specified."

    from src.tf_model import Model
    from tensorflow.python.keras.layers.convolutional import Conv2D
    from tensorflow.python.keras.layers.core import Dense

    model = Model.create_model(args.model_name)
    n_layers = len([1 for layer in model.layers if type(layer) in [Conv2D, Dense]])

    if len(args.contamination_rate) == 1:
        return tuple(args.contamination_rate * n_layers)

    assert len(args.contamination_rate) == n_layers, f"The number of specified values does not align with the number " \
                                                     f"of layers ({len(args.contamination_rate)} != {n_layers})"

    return tuple(args.contamination_rate)
