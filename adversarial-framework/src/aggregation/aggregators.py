from copy import deepcopy

import numpy as np
import logging

class Aggregator:
    """ Aggregation behavior """
    def aggregate(self, global_weights, client_weight_list):
        """

        :type client_weight_list: list[np.ndarray]
        """
        raise NotImplementedError("Subclass")


class FedAvg(Aggregator):

    def __init__(self, lr):
        self.lr = lr

    def aggregate(self, global_weights, client_weight_list):
        """Procedure for merging client weights together with `global_learning_rate`."""
        # return deepcopy(client_weight_list[0]) # Take attacker's
        current_weights = global_weights
        new_weights = deepcopy(current_weights)
        # return new_weights
        update_coefficient = self.lr

        for client in range(0, len(client_weight_list)):
            for layer in range(len(client_weight_list[client])):
                new_weights[layer] = new_weights[layer] + \
                                     update_coefficient * (client_weight_list[client][layer] - current_weights[layer])

        return new_weights


class TrimmedMean(Aggregator):

    def __init__(self, beta, lr):
        """

        :type beta: float fraction of values to truncate
        """
        self.beta = beta
        self.lr = lr

        assert 0 < self.beta < 1/2, "Beta must be between zero and 1/2!"

    def aggregate(self, global_weights, client_weight_list):
        assert self.beta < 0.5, "Beta must be smaller than 0.5!"

        truncate_count = int(self.beta * len(client_weight_list))
        assert len(client_weight_list) - (truncate_count * 2) > 0, "Must be more clients for a given beta!"

        current_weights = global_weights
        new_weights = deepcopy(current_weights)

        # sort by parameter
        accumulator = [np.zeros([*layer.shape, len(client_weight_list)], layer.dtype) for layer in new_weights]
        for client in range(0, len(client_weight_list)):
            for layer in range(len(client_weight_list[client])):
                accumulator[layer][..., client] = client_weight_list[client][layer] - current_weights[layer]

        for layer in range(len(accumulator)):
            accumulator[layer] = np.sort(accumulator[layer], -1)
            if truncate_count > 0:
                accumulator[layer] = accumulator[layer][..., truncate_count:-truncate_count]
            else:
                logging.warning(f"Beta is too low ({self.beta}), trimming no values which means we effectively take the mean.")
            new_weights[layer] = new_weights[layer] + \
                                 self.lr * np.mean(accumulator[layer], -1) * \
                                 len(client_weight_list) # Multiply by list of clients

        return new_weights


def build_aggregator(config):
    aggregator = config.server.aggregator
    lr = config.server.global_learning_rate
    if lr < 0:
        logging.info("Using default global learning rate of n/m")
        lr = config.environment.num_clients / config.environment.num_selected_clients
    else:
        lr = lr

    weight_coefficient = lr / config.environment.num_clients

    from src import aggregation
    cls = getattr(aggregation, aggregator["name"])
    if "args" in aggregator:
        return cls(lr=weight_coefficient, **aggregator["args"])
    else:
        return cls(lr=weight_coefficient)

    # if aggregator.name == "FedAvg":
    #     return FedAvg(weight_coefficient)
    # elif aggregator.name == "TrimmedMean":
    #     return TrimmedMean(config['trimmed_mean_beta'], weight_coefficient)
    # else:
    #     raise NotImplementedError(f"Aggregator {aggregator} not supported!")

