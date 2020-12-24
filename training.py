import os

import numpy as np
from tqdm.notebook import tqdm


def epoch_train(args, model, strategy, ds_train):
    accs, losses = [], []
    for imgs1, labels in tqdm(ds_train, 'train', leave=False, mininterval=2):
        # Train step
        loss, acc = strategy.run(model.train_step,
                                 args=(args.method, args.bsz, imgs1, labels))
        loss = strategy.reduce('SUM', loss, axis=None)
        acc = strategy.reduce('SUM', acc, axis=None)

        # Record
        losses.append(float(loss))
        accs.append(float(acc))

    return accs, losses


def epoch_test(args, model, strategy, ds_test):
    accs = []
    for imgs1, labels in tqdm(ds_test, 'test', leave=False, mininterval=2):
        # Train step
        acc = strategy.run(model.test_step, args=(args.bsz, imgs1, labels))
        acc = strategy.reduce('SUM', acc, axis=None)

        # Record
        accs.append(float(acc))
    return accs


def train(args, model, strategy, ds_train, ds_test):
    all_test_accs, all_train_accs, all_train_losses = [], [], []

    try:
        pbar = tqdm(range(args.epochs), 'epochs', mininterval=2)
        for epoch in pbar:
            # Train
            train_accs, train_losses = epoch_train(args, model, strategy, ds_train)
            model.save_weights(os.path.join(args.out, 'model'))
            all_train_accs.append(train_accs)
            all_train_losses.append(train_losses)

            # Test
            test_accs = epoch_test(args, model, strategy, ds_test)
            all_test_accs.append(test_accs)
            pbar.set_postfix_str(f'{np.mean(train_losses):.3} loss, ' \
                                 f'{np.mean(train_accs):.3} acc, ' \
                                 f'{np.mean(test_accs):.3} test acc', refresh=False)
    except KeyboardInterrupt:
        print('keyboard interrupt caught. ending training early')

    return all_test_accs, all_train_accs, all_train_losses