import argparse

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import mixed_precision

import data
import models
import plots
import training

parser = argparse.ArgumentParser()

parser.add_argument('--epochs', type=int)
parser.add_argument('--bsz', type=int)

parser.add_argument('--lr', type=float)
parser.add_argument('--load', action='store_true')
parser.add_argument('--tsne', action='store_true')

parser.add_argument('--method', choices=['ce', 'supcon', 'supcon-pce'])

parser.add_argument('--out', type=str, default='out/')


def run(args):
    # Mixed precision
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)

    # Strategy
    gpus = tf.config.list_physical_devices('GPU')
    print(f'{len(gpus)} gpus')
    if len(gpus) > 1:
        strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = tf.distribute.get_strategy()
    print(f'using {strategy.__class__.__name__} strategy')

    # Data
    ds_train, ds_test = data.load_datasets(args, strategy)
    if len(gpus) <= 1:
        plots.plot_img_samples(args, ds_train, ds_test)

    # Model and optimizer
    with strategy.scope():
        model = models.ContrastModel(args)
        opt = keras.optimizers.SGD(args.lr, momentum=0.9)
        model.optimizer = mixed_precision.LossScaleOptimizer(opt)

    # Train
    metrics = training.train(args, model, strategy, ds_train, ds_test)
    print(f'finished training. achieved {np.mean(metrics[0][-1]):.3} test accuracy')

    # Plot
    plots.plot_metrics(args, metrics)
    if args.tsne:
        plots.plot_tsne(args, model, ds_test)


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)

    run(args)
