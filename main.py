from tensorflow.keras import optimizers

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

    # Model and optimizer
    with strategy.scope():
        model = models.make_model(args, ds_info['nclass'], ds_info['input_shape'])
        model.compile(optimizers.SGD(args.lr, momentum=0.9), steps_per_execution=args.steps_exec)
        if args.debug:
            model._cnn.summary()

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
