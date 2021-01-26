import os
import shutil

from tensorflow.keras import callbacks
import tensorflow_addons as tfa

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


def compile_model(args, model):
    # Optimizer
    opt = tfa.optimizers.SGDW(args.wd, args.lr, momentum=0.9)

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
