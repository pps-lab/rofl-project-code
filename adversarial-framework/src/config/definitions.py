
from dataclasses import dataclass, MISSING, field
from typing import Optional, Dict, Any, List

from mashumaro import DataClassYAMLMixin

"""
This class defines the configuration schema of the framework.
"""

@dataclass
class Quantization(DataClassYAMLMixin):
    """
    Whether to use (probabilistic) quantization

    """
    type: str = MISSING # "probabilistic" or "deterministic"
    bits: int = MISSING
    frac: int = MISSING


@dataclass
class HyperparameterConfig(DataClassYAMLMixin):
    """
    Config for hyperparameters being tuned in this run.
    """
    args: Dict[str, Any]

    # in the future.. config of values to log and when.


@dataclass
class Environment(DataClassYAMLMixin):
    num_clients: int = MISSING
    num_selected_clients: int = MISSING

    num_malicious_clients: int = MISSING
    experiment_name: str = MISSING

    malicious_client_indices: Optional[List[int]] = None

    attack_frequency: Optional[float] = None # """Frequency of malicious parties being selected. Default is None, for random selection"""

    # these should be removed in the future
    attacker_full_dataset: bool = False # """ Whether the attacker has the full dataset """
    attacker_full_knowledge: bool = False

    load_model: Optional[str] = None
    ignore_malicious_update: bool = False

    print_every: int = 1

    save_updates: bool = False
    save_norms: bool = False
    save_weight_distributions: bool = False
    save_history: bool = False
    save_model_at: List[int] = field(default_factory=lambda: [])

    print_backdoor_eval: bool = False

    seed: int = 0 # randomness seed

    use_config_dir: bool = False # whether to use the config parent dir as the target experiment dir


@dataclass
class Dataset(DataClassYAMLMixin):
    dataset: str = MISSING
    data_distribution: str = MISSING
    number_of_samples: int = -1
    augment_data: bool = False


@dataclass
class FederatedDropout(DataClassYAMLMixin):
    rate: float = 1.0
    all_parameters: bool = True # If set to True, applies dropout on all parameters randomly according to the dropout rate.'
                         #'Applicable only if federated_dropout_rate < 1.0.'
    nonoverlap: bool = False # Each client receives a unique mask that is not overlapped with other clients' masks.
    randommask: bool = False # Enable low rank mode instead of federated dropout, i.e. only mask the uplink.


@dataclass
class Server(DataClassYAMLMixin):
    num_rounds: int = MISSING # Number of training rounds.
    num_test_batches: int = MISSING # Number of client epochs.

    federated_dropout: Optional[FederatedDropout] = None
    aggregator: Optional[Dict[str, Any]] = None
    global_learning_rate: Optional[float] = None

    intrinsic_dimension: int = 1000
    gaussian_noise: float = 0.0


@dataclass
class LearningDecay(DataClassYAMLMixin):
    type: str = MISSING # exponential or boundaries

    # exponential
    decay_steps: Optional[int] = None
    decay_rate: Optional[float] = None

    # boundaries
    decay_boundaries: Optional[List[int]] = None
    decay_values: Optional[List[float]] = None


@dataclass
class TrainingConfig(DataClassYAMLMixin):
    num_epochs: int = MISSING
    batch_size: int = MISSING # Client batch size
    learning_rate: float = MISSING
    decay: Optional[LearningDecay] = None
    optimizer: str = "Adam" # Optimizer
    regularization_rate: Optional[float] = None


@dataclass
class NormBound(DataClassYAMLMixin):
    type: str = MISSING # l2 or linf
    value: float = MISSING
    probability: Optional[float] = None # in case of linf, random clip


@dataclass
class MaliciousConfig(DataClassYAMLMixin):
    objective: Dict[str, Any] = MISSING
    evasion: Optional[Dict[str, Any]] = None
    backdoor: Optional[Dict[str, Any]] = None
    attack_type: Optional[str] = None

    attack_start: Optional[int] = 0
    attack_stop: Optional[int] = 10000000


@dataclass
class ClientConfig(DataClassYAMLMixin):
    model_name: str = MISSING
    benign_training: TrainingConfig = MISSING
    quantization: Optional[Quantization] = None
    malicious: Optional[MaliciousConfig] = None
    optimized_training: bool = True # whether to create the TF training loop
    gaussian_noise: Optional[float] = None
    estimate_other_updates: bool = False
    clip: Optional[NormBound] = None
    model_weight_regularization: Optional[float] = None # weight reg for model (l2)


@dataclass
class Config(DataClassYAMLMixin):
    environment: Environment = MISSING
    server: Server = MISSING
    client: ClientConfig = MISSING
    dataset: Dataset = MISSING
    hyperparameters: Optional[HyperparameterConfig] = None