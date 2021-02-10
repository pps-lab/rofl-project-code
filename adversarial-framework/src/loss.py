import tensorflow as tf
import numpy as np


# Define custom loss
def regularized_loss(local_weights, global_weights, alpha):

    # Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
    def loss(y_true, y_pred):
        cross_entropy_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)(y_true, y_pred)
        weight_norm = 0
        layer_i = 0
        for local_weight_layer in local_weights:
            w = local_weight_layer.weights
            if len(w) > 1:
                global_layer_tensor_w = tf.convert_to_tensor(global_weights[layer_i])
                # global_layer_tensor_b = tf.convert_to_tensor(global_weights[layer_i + 1])
                delta_weight = w[0] - global_layer_tensor_w
                weight_norm += tf.nn.l2_loss(delta_weight)
                layer_i += len(w)

        # print(f"ent {cross_entropy_loss}, w {weight_norm}")

        return alpha * cross_entropy_loss + ((1 - alpha) * tf.math.maximum(0, weight_norm))

    # Return a function
    return loss
