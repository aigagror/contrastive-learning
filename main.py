import os

from tensorflow import keras

import data
import models
import plots
import training
import utils
from training import train


def run(args):
    # Setup
    strategy = utils.setup(args)

    # Data
    ds_train, ds_val, ds_info = data.load_datasets(args)
    if args.debug:
        plots.plot_img_samples(args, ds_train, ds_val)

    # Make and compile model
    with strategy.scope():
        # Model
        if args.load:
            model = keras.models.load_model(os.path.join(args.out, 'model'), compile=(not args.recompile),
                                            custom_objects=models.all_custom_objects)
            print('loaded model')
            if args.recompile:
                training.compile_model(args, model)
                print('recompiled model')
        else:
            model = models.make_model(args, ds_info['nclass'], ds_info['input_shape'])
            training.compile_model(args, model)
            print('starting with new model')

        print(f'{len(model.losses)} regularization losses in this model')

    # Print model information
    model.summary()
    if args.debug:
        keras.utils.plot_model(model, 'out/model.png')

    # Train
    train(args, model, ds_train, ds_val, ds_info)

    # Plot
    if args.loss != 'ce':
        plots.plot_hist_sims(args, strategy, model, ds_val)
    if args.tsne:
        plots.plot_tsne(args, strategy, model, ds_val)


if __name__ == '__main__':
    args = utils.parser.parse_args()
    print(args)

    run(args)
