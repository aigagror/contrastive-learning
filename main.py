import argparse

import pandas as pd
import tensorflow as tf
from tensorflow.keras import mixed_precision

import data
import models
import plots
import training

parser = argparse.ArgumentParser()

# Data
parser.add_argument('--data', choices=['cifar10', 'imagenet'])

# Method
parser.add_argument('--method', choices=['ce', 'supcon', 'supcon-pce'])

# Training
parser.add_argument('--epochs', type=int)
parser.add_argument('--bsz', type=int)
parser.add_argument('--l2_reg', type=float)

# Strategy
parser.add_argument('--tpu', action='store_true')
parser.add_argument('--policy', choices=['mixed_bfloat16', 'float32'])

# Other
parser.add_argument('--load', action='store_true')
parser.add_argument('--tsne', action='store_true')
parser.add_argument('--out', type=str, default='out/')


def setup(args):
    # Logging
    tf.get_logger().setLevel('WARNING')
    pd.options.display.float_format = '{:.3}'.format

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

    for dtype in ['bfloat16', 'float16', 'float32']:
        if dtype in args.policy:
            args.dtype = dtype
            break

    return strategy


def run(args):
    # Setup
    strategy = setup(args)

    # Data
    ds_train, ds_val, nclass = data.load_datasets(args, strategy)
    plots.plot_img_samples(args, ds_train, ds_val)

    # Model and optimizer
    with strategy.scope():
        model = models.ContrastModel(args, nclass)

    # Train
    train_df, val_df = training.train(args, strategy, model, ds_train, ds_val)

    # Plot
    plots.plot_metrics(args, train_df, val_df)
    if args.method.startswith('supcon'):
        plots.plot_hist_sims(args, strategy, model, ds_val)
    if args.tsne:
        plots.plot_tsne(args, strategy, model, ds_val)


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)

    run(args)
