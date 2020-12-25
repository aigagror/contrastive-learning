import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import argparse

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import mixed_precision

import data
import models
import plots
import training

parser = argparse.ArgumentParser()

# Data
parser.add_argument('--data', choices=['cifar10', 'imagenet'])
parser.add_argument('--imagenet-train', type=str)
parser.add_argument('--imagenet-val', type=str)

# Method
parser.add_argument('--method', choices=['ce', 'supcon', 'supcon-pce'])

# Training
parser.add_argument('--epochs', type=int)
parser.add_argument('--bsz', type=int)
parser.add_argument('--lr', type=float)

# Other
parser.add_argument('--load', action='store_true')
parser.add_argument('--tsne', action='store_true')
parser.add_argument('--out', type=str, default='out/')


def get_strategy():
    gpus = tf.config.list_physical_devices('GPU')
    print(f'{len(gpus)} gpus')
    if len(gpus) > 1:
        strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = tf.distribute.get_strategy()
    print(f'using {strategy.__class__.__name__} strategy')
    return strategy


def run(args):
    # Mixed precision
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)

    # Strategy
    strategy = get_strategy()

    # Data
    ds_train, ds_test = data.load_datasets(args, strategy)
    if not isinstance(strategy, tf.distribute.MirroredStrategy):
        plots.plot_img_samples(args, ds_train, ds_test)

    # Model and optimizer
    with strategy.scope():
        model = models.ContrastModel(args)
        opt = keras.optimizers.SGD(args.lr, momentum=0.9)
        model.optimizer = mixed_precision.LossScaleOptimizer(opt)

    # Train
    train_df, test_df = training.train(args, model, strategy, ds_train, ds_test)

    # Plot
    plots.plot_metrics(args, train_df, test_df)
    plots.plot_hist_sims(args, model, ds_test)
    if args.tsne:
        plots.plot_tsne(args, model, ds_test)


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)

    run(args)
