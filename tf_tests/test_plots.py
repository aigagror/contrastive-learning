import unittest

import data
import models
import plots
import utils
import tensorflow as tf


class TestPlots(unittest.TestCase):
    def test_hist(self):
        args = '--data=cifar10 --cnn=small-resnet50v2 ' \
               '--bsz=8 --lr=1e-3 ' \
               '--method=supcon-pce --norm-feats'
        args = utils.parser.parse_args(args.split())
        strategy = tf.distribute.MirroredStrategy(['CPU:0', 'CPU:1'])

        _, ds_val, _ = data.load_datasets(args)

        with strategy.scope():
            model = models.make_model(args, nclass=10, input_shape=[32, 32, 3])

        plots.plot_hist_sims(args, strategy, model, ds_val.take(1))


if __name__ == '__main__':
    unittest.main()
