import unittest

import tensorflow as tf

import data
import data.preprocess
import utils


class TestData(unittest.TestCase):

    def test_data_format(self):
        args = '--data-id=tf_flowers --bsz=8 --loss=supcon'
        args = utils.parser.parse_args(args.split())
        _ = utils.setup(args)

        train_augment_config, val_augment_config = utils.load_augment_configs(args)
        ds, ds_info = data.load_datasets(args.data_id, 'train', shuffle=True, repeat=True,
                                         augment_config=train_augment_config, bsz=args.bsz)

        inputs = next(iter(ds))
        img = inputs['image']

        # Image
        tf.debugging.assert_shapes([
            (img, [8, 224, 224, 3])
        ])
        tf.debugging.assert_type(img, tf.uint8, img.dtype)
        tf.debugging.assert_greater_equal(img, tf.zeros_like(img))
        tf.debugging.assert_less_equal(img, 255 * tf.ones_like(img))
        max_val = tf.reduce_max(img)
        tf.debugging.assert_greater(max_val, tf.ones_like(max_val), 'although not necessary. it is highly unlikely '
                                                                    'that the largest pixel value of an image is at '
                                                                    'most 1.')

        # Label
        label = inputs['label']
        tf.debugging.assert_shapes([(label, [8])])
        tf.debugging.assert_type(label, tf.int64, label.dtype)
        tf.debugging.assert_less_equal(label, tf.cast(ds_info.features['label'].num_classes - 1, label.dtype), label)
        tf.debugging.assert_greater_equal(label, tf.zeros_like(label))

        # Contrast
        contrast = inputs['contrast']
        tf.debugging.assert_shapes([(contrast, [8, 8])])
        tf.debugging.assert_type(contrast, tf.uint8, contrast.dtype)
        tf.debugging.assert_less_equal(contrast, 2 * tf.ones_like(contrast))
        tf.debugging.assert_greater_equal(contrast, tf.zeros_like(contrast))


if __name__ == '__main__':
    unittest.main()
