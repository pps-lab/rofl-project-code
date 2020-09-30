import argparse
from util import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bootstrap simulation")
    parser.add_argument('config_file_path', type=str, help="Configuration file location")
    parser.add_argument('-r', '--reset', action='store_true', help="Redistribute data and recompile the rust crypto library")
    parser.add_argument('-d', '--data', action='store_true', help="Redistribute data")
    parser.add_argument('-c', '--compile', action='store_true', help="Recompile rust crypto library")
    args = parser.parse_args()

    config = ConfigLoader(args.config_file_path, log_summary=False)
    compile = args.compile or args.reset
    distribute_data = args.data or args.reset
    bootstrap(config, compile=compile, distribute_data=distribute_data)