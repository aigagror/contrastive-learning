import unittest

import tensorflow as tf

import data
import models
import training
import utils


class TestTraining(unittest.TestCase):
    def test_compiles(self):
        args = '--data=cifar10 --backbone=affine ' \
               '--bsz=8 --lr=1e-3 ' \
               '--loss=partial-supcon '
        args = utils.parser.parse_args(args.split())
        utils.setup(args)

        ds_train, _, _ = data.load_datasets(args)

        strategy = tf.distribute.MirroredStrategy(['CPU:0', 'CPU:1'])
        with strategy.scope():
            model = models.make_model(args, nclass=10, input_shape=[32, 32, 3])
            training.compile_model(args, model)
        model.fit(ds_train, epochs=1, steps_per_epoch=1)

    def test_lr_schedule(self):
        args = '--warmup 1e-1 5 --lr=5e-1 --lr-decays 30 60 80 '
        args = utils.parser.parse_args(args.split())

        lr_scheduler = training.make_lr_scheduler(args)
        self.assertEqual(lr_scheduler(0, None), 0.1)
        self.assertEqual(lr_scheduler(5, None), 0.5)
        self.assertAlmostEqual(lr_scheduler(30, None), 0.05)
        self.assertAlmostEqual(lr_scheduler(60, None), 0.005)
        self.assertAlmostEqual(lr_scheduler(80, None), 0.0005)

    def test_no_warmup_lr_schedule(self):
        args = '--lr=5e-1 --lr-decays 30 60 80 '
        args = utils.parser.parse_args(args.split())

        lr_scheduler = training.make_lr_scheduler(args)
        self.assertEqual(lr_scheduler(0, None), 0.5)
        self.assertEqual(lr_scheduler(5, None), 0.5)
        self.assertAlmostEqual(lr_scheduler(30, None), 0.05)
        self.assertAlmostEqual(lr_scheduler(60, None), 0.005)
        self.assertAlmostEqual(lr_scheduler(80, None), 0.0005)


if __name__ == '__main__':
    unittest.main()
