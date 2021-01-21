import unittest

import tensorflow as tf

from models import custom_losses


class LossesTest(unittest.TestCase):

    def test_simclr_format(self):
        y = tf.random.normal([32, 32])
        x = tf.random.normal([32, 32])
        loss = custom_losses.SimCLR()(y, x)
        tf.debugging.assert_shapes([
            (loss, [])
        ])

    def test_simclr_eye(self):
        y = tf.random.normal([32, 32])
        x = tf.eye(32)
        loss = custom_losses.SimCLR()(y, x)
        tf.debugging.assert_near(loss, tf.zeros_like(loss), atol=1e-2)

    def test_simclr_rand(self):
        y = tf.random.normal([32, 32])
        x = tf.random.normal([32, 32])
        loss = custom_losses.SimCLR()(y, x)
        tf.debugging.assert_greater_equal(loss, tf.zeros_like(loss))


if __name__ == '__main__':
    unittest.main()
