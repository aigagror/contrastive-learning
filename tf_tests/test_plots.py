import unittest

import data
import models
import plots
import utils
import tensorflow as tf


class TestPlots(unittest.TestCase):
    def setUp(self) -> None:
        args = '--data=cifar10 --cnn=small-resnet50v2 ' \
               '--bsz=8 --lr=1e-3 ' \
               '--method=supcon-pce --norm-feats'
        self.args = utils.parser.parse_args(args.split())
        self.strategy = tf.distribute.MirroredStrategy(['CPU:0', 'CPU:1'])

        _, self.ds_val, _ = data.load_datasets(self.args)

        with self.strategy.scope():
            self.model = models.make_model(self.args, nclass=10, input_shape=[32, 32, 3])

    def test_hist(self):
        plots.plot_hist_sims(self.args, self.strategy, self.model, self.ds_val.take(1))

    def test_tsne(self):
        plots.plot_tsne(self.args, self.strategy, self.model, self.ds_val.take(1))


if __name__ == '__main__':
    unittest.main()
