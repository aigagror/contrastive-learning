import data
import models
import plots
import training
import utils


def run(args):
    # Setup
    strategy = utils.setup(args)

    # Data
    ds_train, ds_val, nclass = data.load_datasets(args, strategy)
    plots.plot_img_samples(args, ds_train, ds_val)

    # Model and optimizer
    with strategy.scope():
        model = models.ContrastModel(args, nclass)

    # Train
    training.train(args, strategy, model, ds_train, ds_val)

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
