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

    # Positive singular losses with various shapes
    def test_losses_format_and_output(self):
        for loss_fn in [custom_losses.SimCLR(), custom_losses.SupCon(), custom_losses.PartialSupCon()]:
            for _ in range(100):
                n = tf.random.uniform([], minval=1, maxval=8, dtype=tf.int32)
                rand_shape = [n, n]
                y = tf.random.uniform(rand_shape)
                x = tf.random.normal(rand_shape)
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

    def test_simclr_distribute_eye(self):
        strategy = tf.distribute.MirroredStrategy(['CPU:0', 'CPU:1'])
        def foo():
            y = tf.random.uniform([4, 4])
            x = 100 * tf.eye(4)
            loss = custom_losses.SimCLR(reduction=tf.keras.losses.Reduction.SUM)(y, x)
            return loss

        loss = strategy.run(foo)
        loss = strategy.reduce('SUM', loss, axis=None)
        tf.debugging.assert_near(loss, tf.zeros_like(loss))

    def test_simclr_distribute_equivalency(self):
        strategy = tf.distribute.MirroredStrategy(['CPU:0', 'CPU:1'])
        global_x = tf.random.normal([4, 4])
        global_y = tf.random.uniform([4, 4])
        def foo():
            loss = custom_losses.SimCLR(reduction=tf.keras.losses.Reduction.SUM)(global_y, global_x) / 2
            return loss

        distributed_loss = strategy.run(foo)
        distributed_loss = strategy.reduce('SUM', distributed_loss, axis=None) / 2

        global_loss = custom_losses.SimCLR()(global_y, global_x)
        tf.debugging.assert_equal(global_loss, distributed_loss)

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
