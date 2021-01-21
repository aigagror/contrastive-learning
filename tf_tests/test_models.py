import unittest

import numpy as np
import tensorflow as tf
from tensorflow import keras

import models
import data
import utils
from models import small_resnet_v2
import main


class TestModel(unittest.TestCase):
    def test_resnet50v2_output_shape(self):
        small_resnet = small_resnet_v2.SmallResNet50V2(include_top=False, input_shape=[32, 32, 3])
        out_shape = small_resnet.output_shape
        self.assertEqual(out_shape, (None, 4, 4, 2048))

    def test_l2_reg(self):
        args = '--data=cifar10 --cnn=small-resnet50v2 --l2-reg=1e-3 ' \
               '--bsz=8 --lr=1e-3 --method=ce '
        args = utils.parser.parse_args(args.split())
        utils.setup(args)

        model = models.make_model(args, nclass=10, input_shape=[32, 32, 3])
        count = 0
        for module in model.submodules:
            for attr in ['kernel_regularizer', 'bias_regularizer']:
                if hasattr(module, attr):
                    count += 1
                    self.assertEqual(getattr(module, attr).l2, np.array(1e-3, dtype=np.float32),
                                     getattr(module, attr).l2)

        # Assert at least 40 modules were regularized
        self.assertGreaterEqual(count, 40)

    def test_no_grad_ce(self):
        args = '--data=cifar10 --cnn=small-resnet50v2 ' \
               '--bsz=8 --lr=1e-3 --method=supcon '
        args = utils.parser.parse_args(args.split())
        utils.setup(args)

        model = models.make_model(args, nclass=10, input_shape=[32, 32, 3])
        with tf.GradientTape() as tape:
            imgs = tf.random.uniform([8, 32, 32, 3])
            imgs2 = tf.random.uniform([8, 32, 32, 3])
            batch_sims = tf.random.uniform([8, 8])
            pred, _ = model({'imgs': imgs, 'imgs2': imgs2, 'batch_sims': batch_sims})

            labels = tf.random.uniform([8])
            loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)(labels, pred)
        grad = tape.gradient(loss, model.trainable_weights)
        num_grads = 0
        for g in grad:
            if g is not None:
                num_grads += 1

        # Only classifer weights and bias should have grads
        self.assertEqual(num_grads, 2)

    def test_partial_supcon_compile(self):
        self.skipTest('takes too long')
        args = '--data=cifar10 --cnn=small-resnet50v2 ' \
               '--bsz=8 --lr=1e-3 ' \
               '--method=supcon-pce --norm-feats'
        args = utils.parser.parse_args(args.split())
        utils.setup(args)

        ds_train, _, _ = data.load_datasets(args)

        model = models.make_model(args, nclass=10, input_shape=[32, 32, 3])
        main.compile_model(args, model)
        model.fit(ds_train, epochs=1, steps_per_epoch=1)


if __name__ == '__main__':
    unittest.main()
