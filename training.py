import os
import shutil

from tensorflow.keras import callbacks


def train(args, model, ds_train, ds_val, ds_info):
    log_dir = os.path.join(args.out, 'logs')
    if not args.load:
        if args.out.startswith('gs://'):
            os.system(f"gsutil -m rm {os.path.join(args.out, '**')}")
        else:
            shutil.rmtree(args.out)
            os.mkdir(args.out)

    def scheduler(epoch, lr):
        if epoch in args.lr_decays:
            return lr * 0.1
        else:
            return lr

    try:
        cbks = [
            callbacks.TensorBoard(log_dir, histogram_freq=1, update_freq=args.update_freq),
            callbacks.LearningRateScheduler(scheduler, verbose=1),
            callbacks.ModelCheckpoint(os.path.join(args.out, 'model'), save_weights_only=True)
        ]

        train_steps, val_steps = args.train_steps, args.val_steps
        if args.steps_exec is not None:
            ds_train, ds_val = ds_train.repeat(), ds_val.repeat()
            if args.train_steps is None:
                train_steps = ds_info['train_size'] // args.bsz
                print(f'steps per execution set and train_steps not specified. setting it to train_size // bsz = {train_steps}')
            if args.val_steps is None:
                val_steps = ds_info['val_size'] // args.bsz
                print(f'steps per execution set and val_steps not specified. setting it to val_size // bsz = {val_steps}')

        model.fit(ds_train, validation_data=ds_val, validation_steps=val_steps,
                  initial_epoch=args.init_epoch, epochs=args.epochs, steps_per_epoch=train_steps,
                  callbacks=cbks)
    except KeyboardInterrupt:
        print('keyboard interrupt caught. ending training early')
