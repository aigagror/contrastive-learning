import os

import tensorflow as tf
from tensorflow.keras import optimizers, callbacks

import data
import models
import plots
import utils
import shutil

def scheduler(epoch, lr):
    if epoch in [30, 60, 80]:
        return lr * tf.math.exp(-0.1)
    else:
        return lr


def run(args):
    # Setup
    strategy = utils.setup(args)

    # Data
    ds_train, ds_val, info = data.load_datasets(args)
    plots.plot_img_samples(args, ds_train, ds_val)

    # Model and optimizer
    with strategy.scope():
        model = models.ContrastModel(args, info['nclass'], info['input_shape'])
        model.compile(optimizers.SGD(args.lr, momentum=0.9), steps_per_execution=args.spe)
        model.cnn.summary()

    # Train
    log_dir = os.path.join(args.out, 'logs')
    if not args.load:
        shutil.rmtree(log_dir, ignore_errors=True)
    try:
        cbks = [
            callbacks.TensorBoard(log_dir, histogram_freq=1, write_images=True, update_freq='batch'),
            callbacks.LearningRateScheduler(scheduler),
            callbacks.ModelCheckpoint(os.path.join(args.out, 'model'), save_weights_only=True)
        ]
        model.fit(ds_train, validation_data=ds_val, validation_steps=args.val_steps,
                  initial_epoch=args.init_epoch, epochs=args.epochs, steps_per_epoch=args.train_steps,
                  callbacks=cbks)
    except KeyboardInterrupt:
        print('keyboard interrupt caught. ending training early')

    # Plot
    plots.plot_hist_sims(args, strategy, model, ds_val)
    if args.tsne:
        plots.plot_tsne(args, strategy, model, ds_val)


if __name__ == '__main__':
    args = utils.parser.parse_args()
    print(args)

    run(args)
