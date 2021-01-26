import unittest

import tensorflow as tf
from tensorflow import keras

import models
import utils
from models import small_resnet_v2


class TestModel(unittest.TestCase):
    def test_resnet50v2_output_shape(self):
        small_resnet = small_resnet_v2.SmallResNet50V2(include_top=False, input_shape=[32, 32, 3])
        out_shape = small_resnet.output_shape
        self.assertEqual(out_shape, (None, 4, 4, 2048))

    def test_no_grad_ce(self):
        args = '--data=cifar10 --model=affine ' \
               '--bsz=8 --lr=1e-3 --loss=supcon '
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


if __name__ == '__main__':
    unittest.main()
