import os

import pandas as pd
from tqdm.auto import tqdm


def epoch_train(args, model, strategy, ds, optimize):
    all_accs, all_ce_losses, all_con_losses = [], [], []
    for imgs1, imgs2, labels in ds:
        # Train step
        step_args = (args.method, args.bsz, imgs1, imgs2, labels, optimize)
        acc, ce_loss, con_loss = strategy.run(model.train_step, args=step_args)
        acc = strategy.reduce('SUM', acc, axis=None)
        ce_loss = strategy.reduce('SUM', ce_loss, axis=None)
        con_loss = strategy.reduce('SUM', con_loss, axis=None)

        # Record
        all_accs.append(float(acc))
        all_ce_losses.append(float(ce_loss))
        all_con_losses.append(float(con_loss))

    return all_accs, all_ce_losses, all_con_losses


def train(args, model, strategy, ds_train, ds_test):
    columns = ['epoch', 'acc', 'ce-loss', 'con-loss']
    train_path = os.path.join(args.out, 'train.csv')
    test_path = os.path.join(args.out, 'test.csv')
    if not args.load:
        # Reset metrics
        pd.DataFrame(columns=columns).to_csv(train_path, index=False)
        pd.DataFrame(columns=columns).to_csv(test_path, index=False)

    try:
        pbar = tqdm(range(1, args.epochs + 1), 'epochs', mininterval=2)
        for epoch in pbar:
            # Train
            train_metrics = epoch_train(args, model, strategy, ds_train, optimize=True)
            train_df = pd.DataFrame(dict(zip(columns, (epoch,) + train_metrics)))
            train_df.to_csv(train_path, mode='a', header=False, index=False)

            # Save weights
            model.save_weights(os.path.join(args.out, 'model'))

            # Test
            test_metrics = epoch_train(args, model, strategy, ds_test, optimize=False)
            test_df = pd.DataFrame(dict(zip(columns, (epoch,) + test_metrics)))
            test_df.to_csv(test_path, mode='a', header=False, index=False)

            # Progress bar
            pbar.set_postfix_str(f'train - {dict(train_df.mean())}, test - {dict(test_df.mean())}', refresh=False)
    except KeyboardInterrupt:
        print('keyboard interrupt caught. ending training early')

    return pd.read_csv(train_path), pd.read_csv(test_path)
