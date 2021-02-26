import os
import tempfile
import unittest

import matplotlib.pyplot as plt
import tensorflow as tf

import utils
from training import lr_schedule, get_lr_scheduler
from tqdm.auto import tqdm

class TestTraining(unittest.TestCase):
    def test_plot_lr_schedules(self):
        plt.figure()
        for args in ['--warmup=5 --lr=5e-1 --train-steps=10 --epochs=90 --cosine-decay ',
                     '--warmup=5 --lr=5e-1 --train-steps=10 --epochs=90 --lr-decays 30 60 80 ']:
            args = utils.parser.parse_args(args.split())

            lr_scheduler = get_lr_scheduler(args)

            x, y = [], []
            for step in tqdm(range(args.train_steps * args.epochs)):
                x.append(step)
                y.append(lr_scheduler(step).numpy())
            plt.plot(x, y, label=f'{args}')
        plt.savefig('out/lr_schedules.jpg')

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

    def test_save_load_lr_schedule(self):
        args = '--warmup=5 --lr=5e-1 --lr-decays 30 60 80 --train-steps=1000'
        args = utils.parser.parse_args(args.split())
        lr_scheduler = lr_schedule.PiecewiseConstantDecayWithWarmup(args.lr, args.train_steps, args.warmup,
                                                                    args.lr_decays)
        optimizer = tf.keras.optimizers.SGD(lr_scheduler)
        inputs = tf.keras.Input([1])
        outputs = tf.keras.layers.Dense(1)(inputs)
        model = tf.keras.Model(inputs, outputs)
        model.compile(optimizer=optimizer)
        model_path = os.path.join(tempfile.gettempdir(), 'model')
        model.save(model_path)
        loaded_model = tf.keras.models.load_model(model_path, custom_objects=utils.all_custom_objects)


if __name__ == '__main__':
    unittest.main()
