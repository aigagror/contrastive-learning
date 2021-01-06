import os
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm.auto import tqdm

import optim


def make_status_str(train_df, val_df):
    train_status = dict(train_df.mean())
    val_status = dict(val_df.drop(columns='epoch').mean())

    now = datetime.now().strftime("%H:%M:%S")

    ret = f'{now} - train: '
    for k, v in train_status.items():
        ret += f'{v:.3} {k}, '
    ret += 'val: '
    for k, v in val_status.items():
        ret += f'{v:.3} {k}, '
    return ret


def get_step_fns(args, model):
    if args.method == 'ce':
        train_step = model.ce_train
        val_step = model.ce_val
    else:
        assert args.method.startswith('supcon')
        train_step = model.supcon_train
        val_step = model.supcon_val
    return train_step, val_step


def epoch_train(args, strategy, step_fn, ds):
    metrics_dict = defaultdict(lambda: [])
    pbar = tqdm(ds)
    for input in pbar:
        # Step
        step_args = input + (tf.cast(args.bsz, args.dtype),)
        step_info = strategy.run(step_fn, args=step_args)

        # Metrics
        status_str = 'loss: '
        for key, value in step_info.items():
            value = float(strategy.reduce('SUM', value, axis=None))
            metrics_dict[key].append(value)
            status_str += f'{value:.3} {key}, '

        pbar.set_postfix_str(status_str, refresh=False)

    return metrics_dict


def get_starting_epoch(args):
    train_path = os.path.join(args.metrics_out, 'train.csv')
    if args.load and os.path.exists(train_path):
        train_df = pd.read_csv(train_path)
        start_epoch = train_df['epoch'].max() + 1
    else:
        start_epoch = 0
    return start_epoch


def train(args, strategy, model, ds_train, ds_val):
    # Setup
    start_epoch = get_starting_epoch(args)

    # Train steps
    train_step_fn, val_step_fn = get_step_fns(args, model)

    # Optimizer
    optim.set_global_optimizer(args)

    # Train
    try:
        for epoch in (start_epoch + np.arange(args.epochs)):
            # Set learning rate
            lr = optim.set_global_lr(args, epoch)

            # Train
            train_metrics = epoch_train(args, strategy, train_step_fn, ds_train)

            # Save weights
            model.save_weights(args.model_out)

            # Validate
            val_metrics = epoch_train(args, strategy, val_step_fn, ds_val)

            # Record metrics
            record_metrics(args, train_metrics, val_metrics, epoch, lr)

    except KeyboardInterrupt:
        print('keyboard interrupt caught. ending training early')

    model.save_weights(args.model_out)
    return


def record_metrics(args, train_metrics, val_metrics, epoch, lr):
    # Determine write configuration
    train_path = os.path.join(args.metrics_out, 'train.csv')
    val_path = os.path.join(args.metrics_out, 'val.csv')
    if epoch == 0 and not args.load:
        mode, header = 'w', True
    else:
        mode, header = 'a', False

    # Record train epoch results
    train_metrics['lr'], train_metrics['epoch'] = lr, epoch
    val_metrics['lr'], val_metrics['epoch'] = lr, epoch
    train_df = pd.DataFrame(train_metrics)
    val_df = pd.DataFrame(val_metrics)

    # Write metrics to disk
    train_df.to_csv(train_path, mode=mode, header=header, index=False)
    val_df.to_csv(val_path, mode='a', header=False, index=False)
    print(make_status_str(train_df, val_df))
