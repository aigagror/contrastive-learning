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
        self.skipTest('legacy')
        args = '--data=cifar10 --backbone=affine ' \
               '--bsz=8 --lr=1e-3 --loss=supcon '
        args = utils.parser.parse_args(args.split())
        utils.setup(args)

        model = models.make_model(args, nclass=10, input_shape=[32, 32, 3])
        with tf.GradientTape() as tape:
            imgs = tf.random.uniform([8, 32, 32, 3])
            imgs2 = tf.random.uniform([8, 32, 32, 3])
            contrast = tf.random.uniform([8, 8])
            pred, _ = model({'imgs': imgs, 'imgs2': imgs2, 'contrast': contrast})

            labels = tf.random.uniform([8])
            loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)(labels, pred)
        grad = tape.gradient(loss, model.trainable_weights)
        num_grads = 0
        for g in grad:
            if g is not None:
                num_grads += 1

        # Only classifer weights and bias should have grads
        self.assertEqual(num_grads, 2)

    def test_equal_proj(self):
        args = '--backbone=affine --loss=supcon '
        args = utils.parser.parse_args(args.split())
        utils.setup(args)

        model = models.make_model(args, nclass=10, input_shape=[32, 32, 3])

        proj_outs = [model.get_layer(name='proj_feats').output, model.get_layer(name='proj_feats2').output]

        proj_model = keras.Model(model.inputs, proj_outs)

        # Equal proj
        inputs = {'imgs': tf.ones([1, 32, 32, 3]), 'imgs2': tf.ones([1, 32, 32, 3])}
        outputs = proj_model(inputs)
        tf.debugging.assert_equal(outputs[0], outputs[1])

        # Unequal proj
        inputs = {'imgs': tf.random.normal([1, 32, 32, 3]), 'imgs2': tf.random.normal([1, 32, 32, 3])}
        outputs = proj_model(inputs)
        tf.debugging.assert_none_equal(outputs[0], outputs[1])

    def test_l2_reg(self):
        args = '--data=cifar10 --backbone=resnet50v2 --weight-decay=1e-3 ' \
               '--bsz=8 --lr=1e-3 --loss=ce '
        args = utils.parser.parse_args(args.split())
        utils.setup(args)

        model = models.make_model(args, nclass=10, input_shape=[32, 32, 3])

        # Assert regularization on at least 40 modules
        self.assertGreaterEqual(len(model.losses), 40)

    def test_num_l2_reg(self):
        for loss, feat_norm, proj_norm, target_reg in [('ce', '', '', 4), ('supcon', '', '', 5),
                                            ('ce', '--feat-norm=bn', '--proj-norm=sn', 2),
                                            ('supcon', '--feat-norm=bn', '--proj-norm=sn', 2)]:
            args = f'--backbone=affine --weight-decay=1e-3 --loss={loss} {feat_norm} {proj_norm}'
            args = utils.parser.parse_args(args.split())
            utils.setup(args)

            model = models.make_model(args, nclass=10, input_shape=[32, 32, 3])

            self.assertEqual(len(model.losses), target_reg, str((loss, feat_norm, target_reg)))

    def test_no_l2_reg(self):
        args = '--data=cifar10 --backbone=affine --weight-decay=0 ' \
               '--bsz=8 --lr=1e-3 --loss=ce '
        args = utils.parser.parse_args(args.split())
        utils.setup(args)

        model = models.make_model(args, nclass=10, input_shape=[32, 32, 3])

        # Assert regularization
        self.assertGreaterEqual(len(model.losses), 0)


if __name__ == '__main__':
    unittest.main()
