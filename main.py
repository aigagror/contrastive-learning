import logging
import os

import tensorflow_datasets as tfds
from tensorflow import keras

import data
import models
import plots
import training
import utils
from training import train
from utils import prepare_tensorboard_dev_logs, set_epoch_steps


def run(args):
    # Setup
    strategy = utils.setup(args)

    # Data
    train_augment_config, val_augment_config = utils.load_augment_configs(args)
    ds_train, ds_info = data.load_datasets(args.data_id, 'train', shuffle=True, repeat=True,
                                           augment_config=train_augment_config, bsz=args.bsz)
    val_split_name = data.get_val_split_name(ds_info)
    ds_val, _ = data.load_datasets(args.data_id, val_split_name, shuffle=False, repeat=False,
                                   augment_config=val_augment_config, bsz=args.bsz)

    # Show examples
    train_fig = tfds.show_examples(ds_train.unbatch(), ds_info, rows=1)
    val_fig = tfds.show_examples(ds_val.unbatch(), ds_info, rows=1)
    train_fig.savefig('out/train_examples.jpg'), val_fig.savefig('out/val_examples.jpg')
    logging.info("dataset examples saved to './out'")

    # Set training and validation steps
    set_epoch_steps(args, ds_info)

    # Make and compile model
    with strategy.scope():
        # Model
        if args.load:
            model = keras.models.load_model(os.path.join(args.out, 'model'), compile=(not args.recompile),
                                            custom_objects=utils.all_custom_objects)
            logging.info('loaded model')
        else:
            model = models.make_model(args, ds_info.features['label'].num_classes, ds_info.features['image'].shape)
            logging.info('starting with new model')

        # Compile?
        if args.recompile or not args.load:
            logging.info('(re)compiling model')
            training.compile_model(args, model)

        logging.info(f'{len(model.losses)} regularization losses in this model')

    # Print model information
    keras.utils.plot_model(model, 'out/model.png')
    logging.info("model graph saved to 'out/model.png'")
    model.summary()

    # Train
    train(args, model, ds_train, ds_val)

    # Plot
    if args.loss != 'ce':
        plots.plot_hist_sims(args, strategy, model, ds_val)
    if args.tsne:
        plots.plot_tsne(args, strategy, model, ds_val)

    # Upload Tensorboard data
    return prepare_tensorboard_dev_logs(args)


if __name__ == '__main__':
    args = utils.parser.parse_args()
    run(args)
