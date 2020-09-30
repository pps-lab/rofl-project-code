# Federated Learning with Adversaries: A Taxonomy of Malicious Behavior, Detection, and Defenses

This framework implements _federated averaging algorithm_, fully described in "[Communication-Efficient Learning of Deep Networks from Decentralized Data
](https://arxiv.org/abs/1602.05629)" paper. It extends its functionality by allowing some clients to misbehave whose aim is to whether introduce a backdoor functionality or impair global model performances. 

### Requirements
- Python 3 (tested on version 3.5)
- [TensorFlow](https://www.tensorflow.org/) (version 2.0)

There is currently no GPU support.

### Installation

#### Setup virtualenv
```bash
virtualenv venv -p python3
source venv/bin/activate
pip install -r common.txt

pip install -r requirements.txt # only required for visualization
```

## Program parameters 

- `num_clients` - Total number of clients.
- `num_selected_clients` - Total number of clients.
- `num_malicious_clients` - Total number of malicious clients.
- `global_learning_rate` - Server's global learning rate, `theta`. By default, this is set to `num_clients/num_selected_clients`, replacing the global model each iteration with the average of the models of the selected clients (old behavior). According to _Bagdasaryan et al._, the learning rate must be lower than this for CIFAR-10 tasks. 
- `attack_type` - Attack type.
- `targeted_deterministic_attack_objective` - All malicious clients try to make the model misclassify a given input as this predefined objective. Only Applicable if num_malicious_clients is non-zero value and 'attack_type' is 'targeted_deterministic'
- `targeted_attack_objective` - Malicious clients try to make the model classify every sample of a class (first argument) as a target (second argument). Only Applicable if num_malicious_clients is non-zero value and 'attack_type' is 'targeted'
- `scale_attack` - Whether malicious clients scale their updates.
- `scale_attack_weight` - A scaling factor for malicious clients' updates. Only applicable if scale_attack is set to True.
- `attack_after` - Start the attack after this amount of rounds. Defaults to 0
- `attack_frequency` - Frequency with which 1 attacker is selected. Dont set for random selection.
- `data_distribution` - IID or non-IID.
- `num_rounds` - Number of training rounds.
- `num_epochs` - Number of client epochs.
- `batch_size` - Clients' batch size.
- `optimizer` - Which optimizer to use.
- `learning_rate` - Learning rate for selected optimizer.
- `decay_steps` - Decay steps for exponential decay. _Optional_.
- `decay_rate` - Decay rate for exponential decay. _Optional_.
- `mal_learning_rate` - Malicious earning rate for selected optimizer.
- `mal_decay_steps` - Malicious decay steps for exponential decay. _Optional_.
- `mal_decay_rate` - Malicious decay rate for exponential decay. _Optional_.
- `mal_num_epochs` - Number of malicious epochs.
- `experiment_name` - Log file path.
- `print_every` - After how many rounds to log test metric.
- `model_name` - Which model to use.
- `seed` - Seed for random functions. Ensures experiment reproducibility.
- `clip` - A positive value for absolute update clipping.
- `clip_probability` - Clip values probabilistically.
- `clip_l2` - Scale layers by maximum l2 norm value.
- `clip_layers` - Indexes of layers to clip. Leave empty for all layers (default)
- `weight_regularization_alpha` - Weight regularization (l2-norm) to use. Keep 1 for none.
- `dataset` - Which dataset to use.
- `workers` - How many threads to use for client training simulation.
- `number_of_samples` - How many samples to use for training; default value of -1 indicates to use full dataset.
- `backdoor_tasks` - The amount of auxiliary datasets of malicious clients to use for the global malicious auxiliary datasets. Applicable if `attack_type` is _backdoor_.
- `aux_samples` - Maximum amount of applicable samples. This is a limit on the result of `backdoor_tasks`, so `backdoor_tasks` should always be set. Applicable if `attack_type` is _backdoor_
- `backdoor_type` - `semantic` for semantic backdoor or `trigger` to include a trigger in the auxiliary images.
- `federated_dropout_rate` - Percentage of neurons (or filters for convolutional layers) that are kept on each layer.
- `keep_history` - Whether Server keeps parameter history. Warning: It slows down the training because of principle eigenvalue computation.
- `pgd` - (Projected Gradient Descent) Weather malicious clients project their gradients onto the feasible set. Compatible with all implemented attacks.
- `pgd_constraint` - Projection bound (applicable only if `pgd` is set).
- `backdoor_feature_aux_train` - Indexes of auxiliary train images. Applicable when `attack_type` is _backdoor_feature_.
- `backdoor_feature_aux_test` - Indexes of auxiliary test images. Applicable when `attack_type` is _backdoor_feature_.
- `backdoor_feature_target` - Target class to flip the malicious auxiliary images to. Applicable when `attack_type` is _backdoor_feature_.
- `poison_samples` - Number of samples to poison in each batch for _data poison_ attack strategy.
- `mal_num_batch` - Amount of batches to run for _data poison_ attack strategy.
- `save_model_at` - What rounds to save the model at.
- `load_model` - Initialize global model with existing model.

## Sample usage:

The entry point of the framework is `main.py` located in `src` directory. 

The example below runs the program with 48 clients, out of which 1 is malicious that performs targeted attack and scales its updates by a factor of 48. See program parameters for more details. 

```bash
python -m src.main --num_clients 48 --num_selected_clients 48 --num_malicious_clients 1 --attack_type targeted --targeted_attack_objective 7 8 --scale_attack True --scale_attack_weight 48 --num_rounds 40 --batch_size 128 --num_epochs 1  --optimizer SGD --learning_rate 0.1 --model dev --clip 0.0015 --experiment_name hist_exp_clipping
```

Program parameters could be also specified in `.yml` file as in the following example:
```bash
python -m src.main -c ./train_configs/federated_dropout.yml --experiment_name fdrop
```

## Visualization sample usage:

The examples below visualize the results from the sample framework run:

### Plot accuracy
```bash
python -m plotting.baseline --experiment_name=hist_exp_clipping --dataset=mnist --samples=-1 --clients 48
```


### Visualize updates

Histogram of local model updates benign vs. adversarial

```bash
python -m visualize.visualize_updates --experiment_name=hist_exp_clipping --model_name=dev
```

### Heat maps

```bash
python -m visualize.heat_map --experiment_name=hist_exp_clipping --model_name=dev
```

## Tensorboard 
Some of the results are displayed in _tensorboard_

```bash
tensorboard --logdir ./experiments/{experiment_name}
```