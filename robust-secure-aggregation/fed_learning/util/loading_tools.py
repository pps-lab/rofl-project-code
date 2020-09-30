import tensorflow as tf

THETA_NAME = "Theta"

def get_theta():
    """ Sadly a very ugly hack to serialize layers with global dependency"""
    var = [v for v in tf.global_variables() if v.name == f"{THETA_NAME}:0"]
    assert len(var) == 1, "Theta was not yet created!"
    return var[0]