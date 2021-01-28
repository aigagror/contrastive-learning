import os

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import callbacks, optimizers

from models import custom_losses


def train(args, model, ds_train, ds_val, ds_info):
    # Callbacks
    cbks = get_callbacks(args)

    try:
        train_steps, val_steps = args.train_steps, args.val_steps
        if args.steps_exec is not None:
            ds_train, ds_val = ds_train.repeat(), ds_val.repeat()
            if args.train_steps is None:
                train_steps = ds_info['train_size'] // args.bsz
                print('steps per execution set and train_steps not specified. '
                      f'setting it to train_size // bsz = {train_steps}')
            if args.val_steps is None:
                val_steps = ds_info['val_size'] // args.bsz
                print('steps per execution set and val_steps not specified. '
                      f'setting it to val_size // bsz = {val_steps}')

        model.fit(ds_train, initial_epoch=args.init_epoch, epochs=args.epochs,
                  validation_data=ds_val, validation_steps=val_steps, steps_per_epoch=train_steps,
                  callbacks=cbks)
    except KeyboardInterrupt:
        print('keyboard interrupt caught. ending training early')


def make_lr_scheduler(args):
    def scheduler(epoch, _):
        # Warmup?
        if args.warmup and epoch < args.warmup[1]:
            return np.linspace(args.warmup[0], args.lr, int(args.warmup[1]))[epoch]

        # Main LR
        curr_lr = args.lr
        if args.lr_decays is None:
            return curr_lr

        # Decay
        for e in range(epoch + 1):
            if e in args.lr_decays:
                curr_lr *= 0.1

        return curr_lr

    return scheduler


def get_callbacks(args):
    cbks = [callbacks.TensorBoard(os.path.join(args.out, 'logs'), update_freq=args.update_freq, write_graph=False)]

    # Save work?
    if not args.no_save:
        cbks.append(callbacks.ModelCheckpoint(os.path.join(args.out, 'model'), verbose=1,
                                              save_best_only=True, monitor='val_loss', mode='min'))

    # Learning rate schedule
    lr_scheduler = make_lr_scheduler(args)
    cbks.append(callbacks.LearningRateScheduler(lr_scheduler, verbose=1))

    return cbks


def compile_model(args, model):
    # Optimizer
    if args.optimizer == 'sgd':
        opt = optimizers.SGD(args.lr, momentum=0.9)
    elif args.optimizer == 'adam':
        opt = optimizers.Adam(args.lr)
    elif args.optimizer == 'lamb':
        opt = tfa.optimizers.LAMB(args.lr)
    else:
        raise Exception(f'unknown optimizer {args.optimizer}')
    if args.debug:
        print(f'{opt.__class__.__name__} optimizer')

    # Loss and metrics
    ce_loss = tf.keras.losses.SparseCategoricalCrossentropy(name='ce', from_logits=True)
    acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='acc')
    ce_metric = tf.keras.metrics.SparseCategoricalCrossentropy(name='ce', from_logits=True)
    losses = {'labels': ce_loss}
    metrics = {'labels': [acc_metric, ce_metric]}

    contrast_loss_dict = {
        'supcon': custom_losses.SupCon(),
        'partial-supcon': custom_losses.PartialSupCon(),
        'simclr': custom_losses.SimCLR(),
        'no-op': custom_losses.NoOp()
    }
    if args.loss in contrast_loss_dict:
        losses['contrast'] = contrast_loss_dict[args.loss]
        if args.model is not None and not args.model.endswith('-norm'):
            print('WARNING: Optimizing over contrastive loss without l2 normalization')

    # Compile
    model.compile(opt, losses, metrics, steps_per_execution=args.steps_exec)
