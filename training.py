import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm.auto import tqdm


def make_status_str(train_df, val_df):
    train_status = dict(train_df.drop(columns='epoch').mean())
    val_status = dict(val_df.drop(columns='epoch').mean())
    ret = 'train: '
    for k, v in train_status.items():
        ret += f'{v:.3} {k}, '
    ret += 'val: '
    for k, v in val_status.items():
        ret += f'{v:.3} {k}, '
    return ret


def epoch_train(args, model, strategy, ds, train):
    all_accs, all_ce_losses, all_con_losses = [], [], []
    pbar = tqdm(ds)
    supcon = args.method == 'supcon'
    step_fn = model.train_step if train else model.test_step
    for imgs1, imgs2, labels in pbar:
        # Step
        step_args = (imgs1, imgs2, labels, tf.cast(args.bsz, args.dtype), supcon)
        acc, ce_loss, con_loss = strategy.run(step_fn, args=step_args)
        acc = strategy.reduce('SUM', acc, axis=None)
        ce_loss = strategy.reduce('SUM', ce_loss, axis=None)
        con_loss = strategy.reduce('SUM', con_loss, axis=None)

        # Record
        all_accs.append(acc)
        all_ce_losses.append(ce_loss)
        all_con_losses.append(con_loss)

        pbar.set_postfix_str(f'{acc:.3} acc, {ce_loss:.3} ce, {con_loss:.3} supcon')

    return all_accs, all_ce_losses, all_con_losses


def train(args, model, strategy, ds_train, ds_val):
    pd.options.display.float_format = '{:.3}'.format
    columns = ['epoch', 'acc', 'ce-loss', 'con-loss']
    train_path = os.path.join(args.out, 'train.csv')
    val_path = os.path.join(args.out, 'val.csv')
    if not args.load:
        # Reset metrics
        pd.DataFrame(columns=columns).to_csv(train_path, index=False)
        pd.DataFrame(columns=columns).to_csv(val_path, index=False)
        start_epoch = 1
    else:
        start_epoch = pd.read_csv(train_path)['epoch'].max() + 1

    try:
        for epoch in (start_epoch + np.arange(args.epochs)):
            # Train
            train_metrics = epoch_train(args, model, strategy, ds_train, train=True)
            train_df = pd.DataFrame(dict(zip(columns, (epoch,) + train_metrics)))
            train_df.to_csv(train_path, mode='a', header=False, index=False)

            # Save weights
            model.save_weights(os.path.join(args.out, 'model'))

            # Validate
            val_metrics = epoch_train(args, model, strategy, ds_val, train=False)
            val_df = pd.DataFrame(dict(zip(columns, (epoch,) + val_metrics)))
            val_df.to_csv(val_path, mode='a', header=False, index=False)

            print(make_status_str(train_df, val_df))
    except KeyboardInterrupt:
        print('keyboard interrupt caught. ending training early')

    return pd.read_csv(train_path), pd.read_csv(val_path)
