import unittest

import tensorflow as tf

import data
import models
import training
import utils
from training import lr_schedule


class TestTraining(unittest.TestCase):
    def test_compiles(self):
        self.skipTest('too long')
        args = '--data=fake-cifar10 --backbone=affine ' \
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
        args = '--warmup=5 --lr=5e-1 --lr-decays 30 60 80 --train-steps=1000'
        args = utils.parser.parse_args(args.split())
        lr_scheduler = lr_schedule.PiecewiseConstantDecayWithWarmup(args.lr, args.train_steps, args.warmup,
                                                                    args.lr_decays)

        for step, tar_lr in [(0, 0), (1, 5e-1 / 5000), (2, 10e-1 / 5000), (5000, 5e-1), (30001, 5e-2), (60001, 5e-3),
                             (80001, 5e-4)]:
            lr = lr_scheduler(step)
            tf.debugging.assert_equal(lr, tf.constant(tar_lr, dtype=lr.dtype), message=f'step {step}: {lr} vs {tar_lr}')

    def test_no_warmup_lr_schedule(self):
        args = '--lr=5e-1 --lr-decays 30 60 80 --train-steps=1000'
        args = utils.parser.parse_args(args.split())
        lr_scheduler = lr_schedule.PiecewiseConstantDecayWithWarmup(args.lr, args.train_steps, args.warmup,
                                                                    args.lr_decays)

        for step, tar_lr in [(0, 5e-1), (1, 5e-1), (2, 5e-1), (5000, 5e-1), (30001, 5e-2), (60001, 5e-3),
                             (80001, 5e-4)]:
            lr = lr_scheduler(step)
            tf.debugging.assert_equal(lr, tf.constant(tar_lr, dtype=lr.dtype), message=f'step {step}: {lr} vs {tar_lr}')


if __name__ == '__main__':
    unittest.main()
