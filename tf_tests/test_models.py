import unittest

import numpy as np

import models
import utils
from models import small_resnet_v2


class TestModel(unittest.TestCase):
    def test_resnet50v2_output_shape(self):
        small_resnet = small_resnet_v2.SmallResNet50V2(include_top=False, input_shape=[32, 32, 3])
        out_shape = small_resnet.output_shape
        self.assertEqual(out_shape, (None, 4, 4, 2048))

    def test_l2_reg(self):
        args = '--data=cifar10 --cnn=small-resnet50v2 --l2-reg=1e-3 ' \
               '--bsz=32 --lr=1e-3 --method=ce '
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


if __name__ == '__main__':
    unittest.main()
