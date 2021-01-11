import os

from tensorflow.keras import optimizers, callbacks

import data
import models
import plots
import utils
import shutil

def scheduler(epoch, lr):
    if epoch in [30, 60, 80]:
        return 0.1 * lr
    else:
        return lr


def run(args):
    # Setup
    strategy = utils.setup(args)

    # Data
    ds_train, ds_val, nclass = data.load_datasets(args)
    plots.plot_img_samples(args, ds_train, ds_val)

    # Model and optimizer
    with strategy.scope():
        model = models.ContrastModel(args, nclass)
        model.compile(optimizers.SGD(args.lr, momentum=0.9), steps_per_execution=args.spe)

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
        model.fit(ds_train, epochs=args.epochs, validation_data=ds_val, callbacks=cbks,
                  steps_per_epoch=args.train_steps, validation_steps=args.val_steps)
    except KeyboardInterrupt:
        print('keyboard interrupt caught. ending training early')

    # Plot
    if args.method.startswith('supcon'):
        plots.plot_hist_sims(args, strategy, model, ds_val)
    if args.tsne:
        plots.plot_tsne(args, strategy, model, ds_val)


if __name__ == '__main__':
    args = utils.parser.parse_args()
    print(args)

    run(args)
