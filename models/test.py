import unittest

import numpy as np
import tensorflow as tf

import data
import models
import utils
from models import small_resnet_v2


class TestModel(unittest.TestCase):
    def test_resnet50v2_output_shape(self):
        small_resnet = small_resnet_v2.SmallResNet50V2(include_top=False, input_shape=[32, 32, 3])
        out_shape = small_resnet.output_shape
        self.assertEqual(out_shape, (None, 4, 4, 512))

    def test_l2_reg(self):
        args = '--data=cifar10 --cnn=small-resnet50v2 --l2-reg=1e-3 ' \
               '--bsz=32 --lr=1e-3 --method=ce '
        args = utils.parser.parse_args(args.split())
        utils.setup(args)

        model = models.ContrastModel(args, nclass=10, input_shape=[32, 32, 3])
        count = 0
        for module in model.submodules:
            for attr in ['kernel_regularizer', 'bias_regularizer']:
                if hasattr(module, attr):
                    count += 1
                    self.assertEqual(getattr(module, attr).l2, np.array(1e-3, dtype=np.float32),
                                     getattr(module, attr).l2)

        # Assert at least 40 modules were regularized
        self.assertGreaterEqual(count, 40)

    def test_norm_feats(self):
        args = '--data=cifar10 --cnn=small-resnet50v2 --method=supcon --norm-feats ' \
               '--bsz=32 --lr=1e-1 --lr-decays 60 120 160 ' \
               '--init-epoch=180 --epochs=180 '
        args = utils.parser.parse_args(args.split())
        utils.setup(args)

        ds_train, ds_val, info = data.load_datasets(args)
        model = models.ContrastModel(args, nclass=10, input_shape=[32, 32, 3])
        input = next(iter(ds_val))
        feats = model.features(input['imgs'])
        proj_feats = model.projection(feats)

        # Assert shapes
        tf.debugging.assert_shapes([
            (feats, ['N', 2048]),
            (proj_feats, ['N', 128]),
        ])

        # Assert norm
        feats_norm = tf.linalg.norm(feats, axis=1)
        proj_norm = tf.linalg.norm(proj_feats, axis=1)
        tf.debugging.assert_near(feats_norm, tf.ones_like(feats_norm))
        tf.debugging.assert_near(proj_norm, tf.ones_like(proj_norm))


if __name__ == '__main__':
    unittest.main()
