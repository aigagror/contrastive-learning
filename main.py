import os

from tensorflow import keras

import data
import models
import plots
import training
import utils
from models import custom_layers, custom_losses
from training import train


def run(args):
    # Setup
    strategy = utils.setup(args)

    # Data
    ds_train, ds_val, ds_info = data.load_datasets(args)
    plots.plot_img_samples(args, ds_train, ds_val)

    # Make and compile model
    with strategy.scope():
        # Model
        if args.load:
            all_custom_objects = {**custom_layers.custom_objects, **custom_losses.custom_objects}
            model = keras.models.load_model(os.path.join(args.out, 'model'), custom_objects=all_custom_objects)
            print('loaded model')
            if args.recompile:
                model = training.compile_model(args, model)
                print('recompiled model')
        else:
            model = models.make_model(args, ds_info['nclass'], ds_info['input_shape'])
            model = training.compile_model(args, model)
            print('starting with new model')

        if args.debug:
            model.summary()

    # Train
    train(args, model, ds_train, ds_val, ds_info)

    # Plot
    plots.plot_hist_sims(args, strategy, model, ds_val)
    if args.tsne:
        plots.plot_tsne(args, strategy, model, ds_val)


if __name__ == '__main__':
    args = utils.parser.parse_args()
    print(args)

    run(args)
