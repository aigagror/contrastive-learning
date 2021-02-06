import logging
import os

from tensorflow import keras

import data
import models
import plots
import training
import utils
from training import train
from utils import prepare_tensorboard_dev_logs


def run(args):
    # Setup
    strategy = utils.setup(args)
    logging.info(args)

    # Data
    ds_train, ds_val, ds_info = data.load_datasets(args)
    plots.plot_img_samples(args, ds_train, ds_val)

    # Set training steps
    if args.train_steps is None:
        args.train_steps = ds_info['train_size'] // args.bsz
        logging.info(f'train_steps not specified. setting it to train_size // bsz = {args.train_steps}')
    if args.val_steps is None:
        args.val_steps = ds_info['val_size'] // args.bsz
        logging.info(f'val_steps not specified. setting it to val_size // bsz = {args.val_steps}')

    # Make and compile model
    with strategy.scope():
        # Model
        if args.load:
            model = keras.models.load_model(os.path.join(args.out, 'model'), compile=(not args.recompile),
                                            custom_objects=utils.all_custom_objects)
            logging.info('loaded model')
            if args.recompile:
                training.compile_model(args, model)
                logging.info('recompiled model')
        else:
            model = models.make_model(args, ds_info['nclass'], ds_info['input_shape'])
            training.compile_model(args, model)
            logging.info('starting with new model')

        logging.info(f'{len(model.losses)} regularization losses in this model')

    # Print model information
    keras.utils.plot_model(model, 'out/model.png')
    logging.info("model plotted to 'out/model.png'")
    model.summary()

    # Train
    train(args, model, ds_train, ds_val)

    # Plot
    if args.loss != 'ce':
        plots.plot_hist_sims(args, strategy, model, ds_val)
    if args.tsne:
        plots.plot_tsne(args, strategy, model, ds_val)

    # Upload Tensorboard data
    prepare_tensorboard_dev_logs(args)


if __name__ == '__main__':
    args = utils.parser.parse_args()
    run(args)
