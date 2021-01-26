import os
import shutil

import tensorflow as tf
from tensorflow.keras import callbacks, optimizers

from models import custom_losses


def train(args, model, ds_train, ds_val, ds_info):
    # Output
    if not args.load:
        if args.out.startswith('gs://'):
            os.system(f"gsutil -m rm {os.path.join(args.out, '**')}")
        else:
            shutil.rmtree(args.out)
            os.mkdir(args.out)

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


def get_callbacks(args):
    cbks = []

    # Save work?
    if not args.no_save:
        cbks.append(callbacks.TensorBoard(os.path.join(args.out, 'logs'), histogram_freq=1,
                                          update_freq=args.update_freq, write_graph=False))
        cbks.append(callbacks.ModelCheckpoint(os.path.join(args.out, 'model')))

    # Learning rate schedule
    def scheduler(epoch, _):
        curr_lr = args.lr
        for e in range(epoch + 1):
            if e in args.lr_decays:
                curr_lr *= 0.1
        return curr_lr

    cbks.append(callbacks.LearningRateScheduler(scheduler, verbose=1))
    return cbks


def add_regularization(args, model):
    def compute_l2_loss():
        l2_layer_losses = [
            tf.nn.l2_loss(w) for w in model.trainable_weights
            if 'gamma' not in w.name and 'beta' not in w.name
        ]
        l2_loss = args.weight_decay * tf.add_n(l2_layer_losses)
        return l2_loss

    return compute_l2_loss


def compile_model(args, model):
    # L2 regularization
    if args.weight_decay is not None:
        if len(model.losses) >= 1:
            print('model already has a regularization loss')
        else:
            model.add_loss(add_regularization(args, model))
            print('added l2 regularization')

    # Optimizer
    opt = optimizers.SGD(args.lr, momentum=0.9)

    # Loss and metrics
    losses = {'labels': custom_losses.Float32SparseCategoricalCrossentropy(from_logits=True)}
    metrics = {'labels': 'acc'}

    contrast_loss_dict = {
        'supcon': custom_losses.SupCon(),
        'partial-supcon': custom_losses.PartialSupCon(),
        'simclr': custom_losses.SimCLR(),
        'no-op': custom_losses.NoOp()
    }
    if args.loss in contrast_loss_dict:
        losses['contrast'] = contrast_loss_dict[args.loss]

    # Compile
    model.compile(opt, losses, metrics, steps_per_execution=args.steps_exec)
