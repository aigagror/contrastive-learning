import os

from tensorflow import keras

import data
import models
import plots
import utils
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
            model = keras.models.load_model(os.path.join(args.out, 'model'), compile=False)
            print('loaded model')
        else:
            model = models.make_model(args, ds_info['nclass'], ds_info['input_shape'])
            print('starting with new model')

        model = models.compile_model(args, model)
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
