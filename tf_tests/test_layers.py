import os
import tempfile
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
            (sims, [8, 16])
        ])

    def test_stand_img(self):
        img = tf.io.decode_image(tf.io.read_file('images/imagenet-sample.jpg'))
        img = tf.expand_dims(img, 0)
        tf.debugging.assert_shapes([
            (img, [1, None, None, 3])
        ])
        out = custom_layers.StandardizeImage()(img)

        mean = tf.reduce_mean(out, axis=[0, 1, 2])
        var = tf.math.reduce_std(out, axis=[0, 1, 2])

        tf.debugging.assert_shapes([
            (mean, [3]),
            (var, [3]),
        ])

        tf.debugging.assert_near(mean, tf.zeros_like(mean), atol=0.5)
        tf.debugging.assert_near(var, tf.ones_like(mean), atol=0.5)

    def test_l2_normalize(self):
        x = tf.random.normal([10000, 32])
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

    def test_configs(self):
        identity_config = custom_layers.Identity(name='foo', dtype=tf.float32).get_config()
        custom_layers.Identity(**identity_config)

        custom_layers.Scale(**custom_layers.Scale(1, name='foo').get_config())

    def test_spec_norm_layer_save_load(self):
        dense = tf.keras.layers.Dense(1)
        sn_dense = custom_layers.SpectralNormalization(dense)

        input = tf.keras.Input([1])
        output = sn_dense(input)

        model = tf.keras.models.Model(input, output)
        model_path = os.path.join(tempfile.gettempdir(), 'model')
        model.save(model_path)
        tf.keras.models.load_model(model_path)


if __name__ == '__main__':
    unittest.main()
