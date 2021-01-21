import unittest

import tensorflow as tf

from models import custom_losses


class LossesTest(unittest.TestCase):

    # Error
    def test_y_true_greater_than_two_error(self):
        for loss in [custom_losses.SimCLR(), custom_losses.SupCon(), custom_losses.PartialSupCon()]:
            x = tf.zeros([3, 3])
            y = 3 * tf.eye(3, dtype=tf.uint8)
            with self.assertRaises(Exception):
                loss(y, x)

    def test_y_true_no_inst(self):
        for loss in [custom_losses.SimCLR(), custom_losses.SupCon(), custom_losses.PartialSupCon()]:
            x = tf.zeros([3, 3])
            y = tf.eye(3, dtype=tf.uint8)
            with self.assertRaises(Exception):
                loss(y, x)

    # Positive singular losses with various shapes
    def test_losses_format_and_output(self):
        for loss_fn in [custom_losses.SimCLR(), custom_losses.SupCon(), custom_losses.PartialSupCon()]:
            for _ in range(200):
                n = tf.random.uniform([], minval=1, maxval=3, dtype=tf.int32)
                d = tf.random.uniform([], maxval=3, dtype=tf.int32)
                rand_shape = [n, n + d]
                y = 2 * tf.eye(rand_shape[0], rand_shape[1], dtype=tf.uint8)
                x = tf.random.normal(rand_shape)
                loss = loss_fn(y, x)
                tf.debugging.assert_greater_equal(loss, tf.zeros_like(loss), f'{loss_fn}\nx={x}\ny={y}')
                tf.debugging.assert_shapes([
                    (loss, [])
                ])

    # SimCLR
    def test_simclr_eye(self):
        y = 2 * tf.eye(3, dtype=tf.uint8)
        x = 100 * tf.eye(3)
        loss = custom_losses.SimCLR()(y, x)
        tf.debugging.assert_near(loss, tf.zeros_like(loss))

    def test_simclr_distribute_eye(self):
        strategy = tf.distribute.MirroredStrategy(['CPU:0', 'CPU:1'])
        global_x = 2 * tf.eye(4, dtype=tf.float32)
        global_y = 2 * tf.eye(4, dtype=tf.uint8)

        def foo():
            replica_context = tf.distribute.get_replica_context()
            id = replica_context.replica_id_in_sync_group
            if id == 0:
                y = global_y[:2]
                x = global_x[:2]
            else:
                y = global_y[2:]
                x = global_x[2:]
            loss = custom_losses.SimCLR(reduction=tf.keras.losses.Reduction.SUM)(y, x) / 2
            return loss

        distributed_loss = strategy.run(foo)
        distributed_loss = strategy.reduce('SUM', distributed_loss, axis=None)
        tf.debugging.assert_near(distributed_loss, tf.zeros_like(distributed_loss))

        global_loss = custom_losses.SimCLR()(global_y, global_x)
        tf.debugging.assert_equal(global_loss, distributed_loss)

    # SupCon
    def test_supcon_eye(self):
        y = 2 * tf.eye(3, dtype=tf.uint8)
        x = 100 * tf.eye(3)
        loss = custom_losses.SupCon()(y, x)
        tf.debugging.assert_near(loss, tf.zeros_like(loss), atol=1e-2)

    # Partial SupCon
    def test_partial_supcon_ignore_inst(self):
        y = tf.constant([[2, 1], [1, 2]], tf.uint8)
        x = 2 * tf.constant([[0, 1], [1, 0]], tf.float32)

        loss = custom_losses.PartialSupCon()(y, x)
        tf.debugging.assert_equal(loss, tf.zeros_like(loss))

    def test_partial_supcon_eye(self):
        y = 2 * tf.eye(3, dtype=tf.uint8)
        for _ in range(100):
            x = tf.random.normal([3, 3])
            loss = custom_losses.PartialSupCon()(y, x)
            tf.debugging.assert_all_finite(loss, 'loss not finite')
            tf.debugging.assert_greater_equal(loss, tf.zeros_like(loss))


if __name__ == '__main__':
    unittest.main()
