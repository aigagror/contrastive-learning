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
            for _ in range(500):
                n = tf.random.uniform([], minval=1, maxval=3, dtype=tf.int32)
                d = tf.random.uniform([], minval=1, maxval=32, dtype=tf.int32)

                y = 2 * tf.eye(n, dtype=tf.int32)
                x = tf.random.uniform([n, 2, d], minval=-1, maxval=1)
                loss = loss_fn(y, x)
                tf.debugging.assert_greater_equal(loss, tf.zeros_like(loss), f'{loss_fn}\nx={x}\ny={y}')
                tf.debugging.assert_shapes([
                    (loss, [])
                ])

    # Distribution equivalancy
    def test_distribute_equivalent(self):
        for LossClass in [custom_losses.SimCLR, custom_losses.SupCon, custom_losses.PartialSupCon]:
            strategy = tf.distribute.MirroredStrategy(['CPU:0', 'CPU:1'])
            global_x = tf.random.normal([4, 2, 32])
            global_y = 2 * tf.eye(4, dtype=tf.int32)

            def foo():
                replica_context = tf.distribute.get_replica_context()
                id = replica_context.replica_id_in_sync_group
                if id == 0:
                    y = global_y[:2]
                    x = global_x[:2]
                else:
                    y = global_y[2:]
                    x = global_x[2:]
                loss = LossClass(reduction=tf.keras.losses.Reduction.SUM)(y, x) / 4
                return loss

            distributed_loss = strategy.run(foo)
            distributed_loss = strategy.reduce('SUM', distributed_loss, axis=None)

            global_loss = LossClass()(global_y, global_x)
            tf.debugging.assert_near(global_loss, distributed_loss, atol=1e-4, message=f'{LossClass}')


if __name__ == '__main__':
    unittest.main()
