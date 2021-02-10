

from .definitions import Config

def load_config(config_name):
    with open(config_name, "rb") as f:
        config = Config.from_yaml(f.read())

    return config
