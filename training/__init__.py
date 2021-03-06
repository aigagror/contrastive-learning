import os

import tensorflow as tf
import tensorflow_addons as tfa
from absl import logging
from tensorflow.keras import callbacks, optimizers

from training import custom_losses, lr_schedule


def train(args, model, ds_train, ds_val):
    # Callbacks
    cbks = get_callbacks(args)

    try:
        model.fit(ds_train, initial_epoch=args.init_epoch, epochs=args.epochs,
                  validation_data=ds_val, validation_steps=args.val_steps, steps_per_epoch=args.train_steps,
                  callbacks=cbks)
    except KeyboardInterrupt:
        logging.info('keyboard interrupt caught. ending training early')

    if args.no_save and (args.epochs - args.init_epoch) > 0:
        logging.info('saving model')
        model.save(os.path.join(args.out, 'model'))


def get_callbacks(args):
    cbks = [callbacks.TensorBoard(os.path.join(args.out, 'logs'), update_freq=args.update_freq, write_graph=False,
                                  profile_batch=args.profile_batch)]

    # Save work?
    if not args.no_save:
        cbks.append(callbacks.ModelCheckpoint(os.path.join(args.out, 'model'), verbose=1,
                                              save_best_only=True, monitor='val_loss', mode='min'))

    return cbks


def compile_model(args, model):
    # LR schedule
    lr_scheduler = get_lr_scheduler(args)

    # Optimizer
    opt = get_optimizer(args, lr_scheduler)
    logging.info(f'{opt.__class__.__name__} optimizer with {lr_scheduler.__class__.__name__} scheduler')

    # Loss and metrics
    ce_loss = tf.keras.losses.SparseCategoricalCrossentropy(name='ce', from_logits=True)
    acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='acc')
    ce_metric = tf.keras.metrics.SparseCategoricalCrossentropy(name='ce', from_logits=True)
    losses = {'label': ce_loss}
    metrics = {'label': [acc_metric, ce_metric]}

    contrast_loss_dict = {
        'supcon': custom_losses.SupCon(args.temp),
        'hiercon': custom_losses.HierCon(args.temp),
        'hiercon2': custom_losses.HierCon2(args.temp),
        'simclr': custom_losses.SimCLR(args.temp),
        'no-op': custom_losses.NoOp()
    }
    if args.loss in contrast_loss_dict:
        losses['contrast'] = contrast_loss_dict[args.loss]
        if args.feat_norm is None:
            logging.warning('optimizing over contrastive loss without any feature normalization')

    # Compile
    model.compile(opt, losses, metrics, steps_per_execution=args.steps_exec)


def get_optimizer(args, lr_scheduler):
    if args.optimizer == 'sgd':
        opt = optimizers.SGD(lr_scheduler, momentum=0.9, nesterov=True)
    elif args.optimizer == 'adam':
        opt = optimizers.Adam(lr_scheduler)
    elif args.optimizer == 'lamb':
        opt = tfa.optimizers.LAMB(lr_scheduler, weight_decay_rate=args.weight_decay)
    else:
        raise Exception(f'unknown optimizer {args.optimizer}')
    return opt


def get_lr_scheduler(args):
    if args.cosine_decay:
        lr_scheduler = lr_schedule.WarmUpAndCosineDecay(args.lr, args.train_steps, args.warmup, args.epochs)
    else:
        lr_scheduler = lr_schedule.PiecewiseConstantDecayWithWarmup(args.lr, args.train_steps, args.warmup,
                                                                    args.lr_decays,
                                                                    start_step=args.init_epoch * args.train_steps)
    return lr_scheduler
