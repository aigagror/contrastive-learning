import unittest

import tensorflow as tf

from models import custom_layers


class LayersTest(unittest.TestCase):

    def simple_bsz_sims_use(self):
        a = tf.random.normal([8, 32])
        b = tf.random.normal([8, 32])
        sims = custom_layers.GlobalBatchSims()((a, b))
        return sims

    def test_global_bsz_sims(self):
        sims = self.simple_bsz_sims_use()
        tf.debugging.assert_shapes([
            (sims, [8, 8])
        ])

    def test_distribute_global_bsz_sims(self):
        strategy = tf.distribute.MirroredStrategy(['CPU:0', 'CPU:1'])

        sims = strategy.run(self.simple_bsz_sims_use)
        sims = strategy.reduce('SUM', sims, axis=None)

        tf.debugging.assert_shapes([
            (sims, [16, 16])
        ])

    def test_stand_img(self):
        x = tf.random.uniform([8, 32, 32, 3], maxval=255)
        x = tf.cast(x, tf.uint8)
        y = custom_layers.StandardizeImage()(x)
        tf.debugging.assert_greater_equal(y, -tf.ones_like(y))
        tf.debugging.assert_less_equal(y, tf.ones_like(y))

    def test_l2_normalize(self):
        x = tf.random.normal([8, 32])
        y = custom_layers.L2Normalize()(x)

        # Assert one norm
        y_norm = tf.linalg.norm(y, axis=1)
        tf.debugging.assert_near(y_norm, tf.ones_like(y_norm))

        # Assert same shape
        tf.debugging.assert_shapes([
            (x, ['N', 'M']),
            (y, ['N', 'M']),
        ])

        # Assert equal sign
        prod = x * y
        tf.debugging.assert_greater_equal(prod, tf.zeros_like(prod))


if __name__ == '__main__':
    unittest.main()
