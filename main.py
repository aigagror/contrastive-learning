from tensorflow.keras import optimizers, losses

import data
import models
import plots
import utils


def run(args):
    # Setup
    strategy = utils.setup(args)

    # Data
    ds_train, ds_val, nclass = data.load_datasets(args)
    plots.plot_img_samples(args, ds_train, ds_val)

    # Model and optimizer
    with strategy.scope():
        model = models.ContrastModel(args, nclass)
        model.compile(optimizers.SGD(args.lr, momentum=0.9), metrics=['acc'], steps_per_execution=50)

    # Train
    try:
        model.fit(ds_train, epochs=args.epochs, validation_data=ds_val, callbacks=[])
    except KeyboardInterrupt:
        print('keyboard interrupt caught. ending training early')

    # Plot
    plots.plot_metrics(args)
    if args.method.startswith('supcon'):
        plots.plot_hist_sims(args, strategy, model, ds_val)
    if args.tsne:
        plots.plot_tsne(args, strategy, model, ds_val)


if __name__ == '__main__':
    args = utils.parser.parse_args()
    print(args)

    run(args)
