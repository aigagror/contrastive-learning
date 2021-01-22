import argparse

import tensorflow as tf
from tensorflow.keras import mixed_precision

parser = argparse.ArgumentParser()

# Data
parser.add_argument('--data', choices=['cifar10', 'imagenet'])

# CNN
parser.add_argument('--cnn', choices=['small-resnet50v2', 'resnet50v2'])
parser.add_argument('--norm-feats', action='store_true')

# Method
parser.add_argument('--method', choices=['ce', 'supcon', 'mse-supcon', 'bce-supcon', 'partial-supcon', 'simclr'])

# Training hyperparameters
parser.add_argument('--init-epoch', type=int, default=0)
parser.add_argument('--epochs', type=int)
parser.add_argument('--bsz', type=int)
parser.add_argument('--lr', type=float)
parser.add_argument('--lr-decays', type=int, nargs='+', help='decays learning rate at the specified epochs')
parser.add_argument('--l2-reg', type=float, default=1e-4)
parser.add_argument('--steps-exec', type=int, help='steps per execution')
parser.add_argument('--train-steps', type=int, help='train steps per epoch')
parser.add_argument('--val-steps', type=int, help='val steps per epoch')

# Strategy
parser.add_argument('--tpu', action='store_true')
parser.add_argument('--policy', choices=['mixed_bfloat16', 'float32'], default='float32')

# Other
parser.add_argument('--load', action='store_true')
parser.add_argument('--tsne', action='store_true')
parser.add_argument('--out', type=str, default='out/')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--no-checkpoint', action='store_true')

# Tensorboard
parser.add_argument('--update-freq', type=str, default='epoch', help='tensorboard metrics update frequency')


def setup(args):
    # Logging
    tf.get_logger().setLevel('WARNING')

    # Strategy
    if args.tpu:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(resolver)
        # This is the TPU initialization code that has to be at the beginning.
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.TPUStrategy(resolver)
    elif len(tf.config.list_physical_devices('GPU')) > 1:
        strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = tf.distribute.get_strategy()

    # Mixed precision
    policy = mixed_precision.Policy(args.policy)
    mixed_precision.set_global_policy(policy)

    return strategy
