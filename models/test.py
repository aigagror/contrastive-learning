import unittest

from models import small_resnet_v2


class TestSmallResnet(unittest.TestCase):
    def test_output_shape(self):
        small_resnet = small_resnet_v2.SmallResNet50V2(include_top=False, input_shape=[32, 32, 3])
        out_shape = small_resnet.output_shape
        self.assertEqual(out_shape, (None, 4, 4, 512))


if __name__ == '__main__':
    unittest.main()
