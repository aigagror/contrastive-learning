from absl import logging
import os

import tensorflow_datasets as tfds
from tensorflow import keras

import models
import plots
import training
import utils
from data import load_distributed_datasets
from training import train


def run(args):
    # Setup
    strategy = utils.setup(args)

    # Data
    _, ds_info = tfds.load(args.data_id, try_gcs=True, data_dir='gs://aigagror/datasets', with_info=True)
    train_augment_config, val_augment_config = utils.load_augment_configs(args)
    ds_train, ds_val = load_distributed_datasets(args, ds_info, strategy, train_augment_config, val_augment_config)

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
    plots.plot_hist_sims(args, strategy, model, ds_val)
    if args.tsne:
        plots.plot_instance_tsne(args, strategy, model, ds_val)
        plots.plot_tsne(args, strategy, model, ds_val)

    # Upload Tensorboard data
    return utils.prepare_tensorboard_dev_logs(args)


if __name__ == '__main__':
    args = utils.parser.parse_args()
    run(args)
