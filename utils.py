import argparse
import os
import shutil

import tensorflow as tf
from tensorflow.keras import mixed_precision

parser = argparse.ArgumentParser()

# Data
parser.add_argument('--data', choices=['imagenet', 'cifar10', 'fake-cifar10'])
parser.add_argument('--augment', choices=['auto'])

# Model
parser.add_argument('--backbone', choices=['small-resnet50v2', 'resnet50v2', 'resnet50', 'affine'])
parser.add_argument('--feat-norm', choices=['l2', 'sn'])

# Loss objective
parser.add_argument('--loss', choices=['ce', 'supcon', 'partial-supcon', 'simclr', 'no-op'])
parser.add_argument('--temp', type=float, default=0.1)
parser.add_argument('--weight-decay', type=float, default=1e-4)

# Training hyperparameters
parser.add_argument('--optimizer', choices=['sgd', 'adam', 'lamb'], default='sgd')

parser.add_argument('--init-epoch', type=int, default=0)
parser.add_argument('--epochs', type=int)
parser.add_argument('--steps-exec', type=int, help='steps per execution')
parser.add_argument('--train-steps', type=int, help='train steps per epoch')
parser.add_argument('--val-steps', type=int, help='val steps per epoch')

parser.add_argument('--bsz', type=int)
parser.add_argument('--warmup', type=float, nargs=2)
parser.add_argument('--lr', type=float)
parser.add_argument('--lr-decays', type=int, nargs='+', help='decays learning rate at the specified epochs')

parser.add_argument('--recompile', action='store_true')

# Strategy
parser.add_argument('--tpu', action='store_true')
parser.add_argument('--multi-cpu', action='store_true')
parser.add_argument('--policy', choices=['mixed_bfloat16', 'float32'], default='float32')

# Other
parser.add_argument('--load', action='store_true')
parser.add_argument('--tsne', action='store_true')
parser.add_argument('--base-dir', type=str, default='out/')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--no-save', action='store_true', help='skip saving logs and model checkpoints')
parser.add_argument('--profile-batch', type=int, nargs='*', default=0)

# Tensorboard
parser.add_argument('--update-freq', type=str, default='epoch', help='tensorboard metrics update frequency')


def setup(args):
    # Logging
    tf.get_logger().setLevel('DEBUG' if args.debug else 'WARNING')

    # Output directory
    args.out = os.path.join(args.out, f'{args.data}-{args.backbone}-{args.feat_norm}-{args.loss}')
    if not args.load:
        if args.out.startswith('gs://'):
            os.system(f"gsutil -m rm {os.path.join(args.out, '**')}")
        else:
            shutil.rmtree(args.out)
            os.mkdir(args.out)

    # Strategy
    if args.tpu:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(resolver)
        # This is the TPU initialization code that has to be at the beginning.
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.TPUStrategy(resolver)
    elif len(tf.config.list_physical_devices('GPU')) > 1:
        strategy = tf.distribute.MirroredStrategy()
    elif args.multi_cpu:
        strategy = tf.distribute.MirroredStrategy(['CPU:0', 'CPU:1'])
    else:
        strategy = tf.distribute.get_strategy()

    # Mixed precision
    policy = mixed_precision.Policy(args.policy)
    mixed_precision.set_global_policy(policy)

    return strategy
