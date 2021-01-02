import os

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def make_status_str(train_df, val_df):
    train_status = dict(train_df.mean())
    val_status = dict(val_df.drop(columns='epoch').mean())
    ret = 'train: '
    for k, v in train_status.items():
        ret += f'{v:.3} {k}, '
    ret += 'val: '
    for k, v in val_status.items():
        ret += f'{v:.3} {k}, '
    return ret


def get_train_steps(args, model):
    if args.method == 'ce':
        train_step = model.ce_train
        val_step = model.ce_val
    else:
        assert args.method.startswith('supcon')
        train_step = model.supcon_train
        val_step = model.supcon_val
    return train_step, val_step


def set_metric_dfs(args, columns):
    train_path = os.path.join(args.out, 'train.csv')
    val_path = os.path.join(args.out, 'val.csv')
    if not args.load:
        # Reset metrics
        pd.DataFrame(columns=columns).to_csv(train_path, index=False)
        pd.DataFrame(columns=columns).to_csv(val_path, index=False)
        start_epoch = 1
    else:
        start_epoch = pd.read_csv(train_path)['epoch'].max() + 1
    return start_epoch, train_path, val_path


def epoch_train(args, strategy, step_fn, ds):
    all_accs, all_ce_losses, all_con_losses = [], [], []
    pbar = tqdm(ds)
    for input in pbar:
        # Step
        step_args = input + (float(args.bsz), )
        acc, ce_loss, con_loss = strategy.run(step_fn, args=step_args)
        acc = strategy.reduce('SUM', acc, axis=None)
        ce_loss = strategy.reduce('SUM', ce_loss, axis=None)
        con_loss = strategy.reduce('SUM', con_loss, axis=None)

        # Record
        all_accs.append(float(acc))
        all_ce_losses.append(float(ce_loss))
        all_con_losses.append(float(con_loss))

        pbar.set_postfix_str(f'{acc:.3} acc, {ce_loss:.3} ce, {con_loss:.3} supcon')

    return all_accs, all_ce_losses, all_con_losses


def train(args, strategy, model, ds_train, ds_val):
    # Metrics setup
    columns = ['epoch', 'acc', 'ce-loss', 'con-loss']
    start_epoch, train_path, val_path = set_metric_dfs(args, columns)

    # Train steps
    train_step, val_step = get_train_steps(args, model)

    # Train
    try:
        for epoch in (start_epoch + np.arange(args.epochs)):
            # Train
            train_metrics = epoch_train(args, strategy, train_step, ds_train)
            train_df = pd.DataFrame(dict(zip(columns, (epoch,) + train_metrics)))
            train_df.to_csv(train_path, mode='a', header=False, index=False)

            # Save weights
            model.save_weights(os.path.join(args.out, 'model'))

            # Validate
            val_metrics = epoch_train(args, strategy, val_step, ds_val)
            val_df = pd.DataFrame(dict(zip(columns, (epoch,) + val_metrics)))
            val_df.to_csv(val_path, mode='a', header=False, index=False)

            print(make_status_str(train_df, val_df))
    except KeyboardInterrupt:
        print('keyboard interrupt caught. ending training early')

    model.save_weights(os.path.join(args.out, 'model'))
    return pd.read_csv(train_path), pd.read_csv(val_path)
