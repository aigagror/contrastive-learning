import os
import shutil

from tensorflow.keras import callbacks


def train(args, model, ds_train, ds_val):
    log_dir = os.path.join(args.out, 'logs')
    if not args.load:
        shutil.rmtree(log_dir, ignore_errors=True)

    def scheduler(epoch, lr):
        if epoch in args.lr_decays:
            return lr * 0.1
        else:
            return lr

    try:

        cbks = [
            callbacks.TensorBoard(log_dir, histogram_freq=1, write_images=True, update_freq=args.update_freq),
            callbacks.LearningRateScheduler(scheduler, verbose=1),
            callbacks.ModelCheckpoint(os.path.join(args.out, 'model'), save_weights_only=True)
        ]

        model.fit(ds_train, validation_data=ds_val, validation_steps=args.val_steps,
                  initial_epoch=args.init_epoch, epochs=args.epochs, steps_per_epoch=args.train_steps,
                  callbacks=cbks)
    except KeyboardInterrupt:
        print('keyboard interrupt caught. ending training early')
