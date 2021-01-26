import os
import shutil
import tempfile

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks, optimizers

from models import custom_layers, custom_losses


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


def add_regularization(model, regularizer):
    if not isinstance(regularizer, tf.keras.regularizers.Regularizer):
        print("Regularizer must be a subclass of tf.keras.regularizers.Regularizer")
        return model

    for module in model.submodules:
        for attr in ['kernel_regularizer', 'bias_regularizer']:
            if hasattr(module, attr):
                setattr(module, attr, regularizer)

    # When we change the layers attributes, the change only happens in the model config file
    model_json = model.to_json()

    # Save the weights before reloading the model.
    tmp_weights_path = os.path.join(tempfile.gettempdir(), 'tmp_weights.h5')
    model.save_weights(tmp_weights_path)

    # load the model from the config
    model = tf.keras.models.model_from_json(model_json, custom_objects=custom_layers.custom_objects)

    # Reload the model weights
    model.load_weights(tmp_weights_path, by_name=True)
    return model


def compile_model(args, model):
    # L2 regularization
    if args.l2_reg is not None:
        regularizer = keras.regularizers.l2(args.l2_reg)
        print(f'{args.l2_reg:.3} l2 reg')
    else:
        regularizer = None
        print('no l2 regularization')
    model = add_regularization(model, regularizer)

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

    return model
