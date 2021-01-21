import unittest

import tensorflow as tf

from models import custom_losses


class LossesTest(unittest.TestCase):

    # Error
    def test_y_true_neg_error(self):
        for loss in [custom_losses.SimCLR(), custom_losses.SupCon(), custom_losses.PartialSupCon()]:
            x = tf.zeros([3, 3])
            y = -tf.random.uniform([3, 3])
            with self.assertRaises(tf.errors.InvalidArgumentError):
                loss(y, x)

    def test_y_true_greater_than_one_error(self):
        for loss in [custom_losses.SimCLR(), custom_losses.SupCon(), custom_losses.PartialSupCon()]:
            x = tf.zeros([3, 3])
            y = 1 + tf.random.uniform([3, 3])
            with self.assertRaises(tf.errors.InvalidArgumentError):
                loss(y, x)

    # Positive singular losses
    def test_losses_format_and_output(self):
        for loss_fn in [custom_losses.SimCLR(), custom_losses.SupCon(), custom_losses.PartialSupCon()]:
            for _ in range(100):
                y = tf.random.uniform([3, 3])
                x = tf.random.normal([3, 3])
                loss = loss_fn(y, x)
                tf.debugging.assert_greater_equal(loss, tf.zeros_like(loss), f'{loss_fn}\nx={x}\ny={y}')
                tf.debugging.assert_shapes([
                    (loss, [])
                ])

    # SimCLR
    def test_simclr_eye(self):
        y = tf.random.uniform([3, 3])
        x = 100 * tf.eye(3)
        loss = custom_losses.SimCLR()(y, x)
        tf.debugging.assert_near(loss, tf.zeros_like(loss))

    # SupCon
    def test_supcon_eye(self):
        y = tf.eye(3)
        x = 100 * tf.eye(3)
        loss = custom_losses.SupCon()(y, x)
        tf.debugging.assert_near(loss, tf.zeros_like(loss), atol=1e-2)

    # Partial SupCon
    def test_partial_supcon_ignore_diag(self):
        y1 = tf.constant([[0, 1], [1, 0]])
        y2 = tf.ones([2, 2])

        x = tf.random.normal([2, 2])

        loss1 = custom_losses.PartialSupCon()(y1, x)
        loss2 = custom_losses.PartialSupCon()(y2, x)
        tf.debugging.assert_equal(loss1, loss2)

    def test_partial_supcon_eye(self):
        y = tf.eye(3)
        for _ in range(100):
            x = tf.random.normal([3, 3])
            loss = custom_losses.PartialSupCon()(y, x)
            tf.debugging.assert_all_finite(loss, 'loss not finite')
            tf.debugging.assert_greater_equal(loss, tf.zeros_like(loss))


if __name__ == '__main__':
    unittest.main()
