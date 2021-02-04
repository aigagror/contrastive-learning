import unittest

import tensorflow as tf

import data
import models
import plots
import utils


class TestPlots(unittest.TestCase):
    def basic_usage(self):
        self.skipTest('too long')
        args = '--data=cifar10 --backbone=affine ' \
               '--bsz=8 --lr=1e-3 ' \
               '--loss=partial-supcon '
        args = utils.parser.parse_args(args.split())
        utils.setup(args)
        strategy = tf.distribute.MirroredStrategy(['CPU:0', 'CPU:1'])

        _, ds_val, _ = data.load_datasets(args)

        with strategy.scope():
            model = models.make_model(args, nclass=10, input_shape=[32, 32, 3])

        return args, strategy, model, ds_val

    def test_hist(self):
        args, strategy, model, ds_val = self.basic_usage()
        plots.plot_hist_sims(args, strategy, model, ds_val.take(1))
        logging.info(f'hist saved to {args.out}')

    def test_tsne(self):
        args, strategy, model, ds_val = self.basic_usage()
        plots.plot_tsne(args, strategy, model, ds_val.take(1))
        logging.info(f'tsne saved to {args.out}')


if __name__ == '__main__':
    unittest.main()
