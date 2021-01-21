import os

from tensorflow import keras
from tensorflow.keras import optimizers

import data
import models
import plots
import utils
from models import custom_losses
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
            model = keras.models.load_model(os.path.join(args.out, 'model'))
        else:
            model = models.make_model(args, ds_info['nclass'], ds_info['input_shape'])

        compile_model(args, model)
        if args.debug:
            model.summary()

    # Train
    train(args, model, ds_train, ds_val, ds_info)

    # Plot
    plots.plot_hist_sims(args, strategy, model, ds_val)
    if args.tsne:
        plots.plot_tsne(args, strategy, model, ds_val)


def compile_model(args, model):
    # Optimizer
    opt = optimizers.SGD(args.lr, momentum=0.9)

    # Loss and metrics
    losses = {'labels': keras.losses.SparseCategoricalCrossentropy(from_logits=True)}
    metrics = {'labels': 'acc'}
    if args.method == 'supcon':
        losses['batch_sims'] = custom_losses.SupCon()
        metrics['batch_sims'] = custom_losses.SupCon()
    elif args.method == 'supcon-pce':
        losses['batch_sims'] = [custom_losses.SimCLR(), custom_losses.PartialSupCon()]
        metrics['batch_sims'] = [custom_losses.SimCLR(), custom_losses.PartialSupCon()]

    # Compile
    model.compile(opt, losses, metrics, steps_per_execution=args.steps_exec)


if __name__ == '__main__':
    args = utils.parser.parse_args()
    print(args)

    run(args)
