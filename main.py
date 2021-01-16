import argparse
from source.tools import Trainer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='config.json', type=str)

    return  parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    Trainer(args.config_path)