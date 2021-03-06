import os

import tensorflow as tf
import tensorflow_datasets as tfds
from absl import logging
from tensorflow import keras

import models
import plots
import training
import utils
from data import load_distributed_datasets, get_val_split_name
from training import train


def run(args):
    # Setup
    strategy = utils.setup(args)

    # Data
    _, ds_info = tfds.load(args.data_id, try_gcs=True, data_dir='gs://aigagror/datasets', with_info=True)
    train_augconfig, val_augconfig = utils.load_augment_configs(args)
    val_split_name = get_val_split_name(ds_info)

    ds_train = load_distributed_datasets(args, strategy, ds_info, 'train', train_augconfig, shuffle=True)
    ds_val = load_distributed_datasets(args, strategy, ds_info, val_split_name, val_augconfig)

    # Set training and validation steps
    utils.set_epoch_steps(args, ds_info)

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
    local_strategy = tf.distribute.get_strategy()
    local_ds_val = load_distributed_datasets(args, local_strategy, ds_info, val_split_name, val_augconfig)
    plots.plot_hist_sims(args, strategy, model, ds_val)
    if args.tsne:
        plots.plot_instance_tsne(args, model, local_ds_val)
        plots.plot_tsne(args, strategy, model, ds_val)

    # Upload Tensorboard data
    return utils.prepare_tensorboard_dev_logs(args)


if __name__ == '__main__':
    args = utils.parser.parse_args()
    run(args)
