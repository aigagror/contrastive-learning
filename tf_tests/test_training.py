import unittest

import tensorflow as tf

import data
import models
import training
import utils


class TestTraining(unittest.TestCase):
    def test_compiles(self):
        for loss in ['ce', 'supcon', 'partial-supcon']:
            args = '--data=cifar10 --model=affine ' \
                   '--bsz=8 --lr=1e-3 ' \
                   f'--loss={loss} '
            args = utils.parser.parse_args(args.split())
            utils.setup(args)

            ds_train, _, _ = data.load_datasets(args)

            strategy = tf.distribute.MirroredStrategy(['CPU:0', 'CPU:1'])
            with strategy.scope():
                model = models.make_model(args, nclass=10, input_shape=[32, 32, 3])
                training.compile_model(args, model)
            model.fit(ds_train, epochs=1, steps_per_epoch=1)


if __name__ == '__main__':
    unittest.main()
