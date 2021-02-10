from typing import Tuple, Any

import configargparse
import logging
import src.config as cnf
from src.config.definitions import Config

parser = configargparse.ArgumentParser()
parser.add('-c', '--config_filepath', required=True, help='Path to config file.')

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

def get_config() -> Tuple[Config, Any]:
    args = parser.parse_args()
    config = cnf.load_config(args.config_filepath)

    logging.basicConfig(level=args.loglevel)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    logging.info(config)

    return config, args

