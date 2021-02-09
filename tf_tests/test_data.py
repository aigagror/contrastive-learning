import logging
import unittest

import tensorflow as tf
import tensorflow_datasets as tfds

import data
import data.preprocess
import utils


class TestData(unittest.TestCase):

    def test_augmentation(self):
        args = '--data-id=tf_flowers --autoaugment --bsz=8 --loss=supcon '
        args = utils.parser.parse_args(args.split())
        strategy = utils.setup(args)

        _, ds_info = tfds.load(args.data_id, try_gcs=True, data_dir='gs://aigagror/datasets', with_info=True)
        train_augment_config, val_augment_config = utils.load_augment_configs(args)
        ds_train, ds_val = data.load_distributed_datasets(args, ds_info, strategy, train_augment_config,
                                                          val_augment_config)

        train_fig = tfds.show_examples(ds_train.unbatch(), ds_info, rows=1)
        val_fig = tfds.show_examples(ds_val.unbatch(), ds_info, rows=1)
        train_fig.savefig('out/train_examples.jpg'), val_fig.savefig('out/val_examples.jpg')
        logging.info("dataset examples saved to './out'")

    def test_data_format(self):
        args = '--data-id=tf_flowers --bsz=8 --loss=supcon'
        args = utils.parser.parse_args(args.split())
        _ = utils.setup(args)

        _, ds_info = tfds.load(args.data_id, try_gcs=True, data_dir='gs://aigagror/datasets', with_info=True)
        train_augment_config, val_augment_config = utils.load_augment_configs(args)
        input_ctx = tf.distribute.InputContext()
        ds = data.source_dataset(input_ctx, ds_info, args.data_id, 'train', args.cache, shuffle=True, repeat=True,
                                 augment_config=train_augment_config, global_bsz=args.bsz)

        inputs, targets = next(iter(ds))
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
        label = targets['label']
        tf.debugging.assert_shapes([(label, [8])])
        tf.debugging.assert_type(label, tf.int64, label.dtype)
        tf.debugging.assert_less_equal(label, tf.cast(ds_info.features['label'].num_classes - 1, label.dtype), label)
        tf.debugging.assert_greater_equal(label, tf.zeros_like(label))

        # Contrast
        contrast = targets['contrast']
        tf.debugging.assert_shapes([(contrast, [8, 8])])
        tf.debugging.assert_type(contrast, tf.uint8, contrast.dtype)
        tf.debugging.assert_less_equal(contrast, 2 * tf.ones_like(contrast))
        tf.debugging.assert_greater_equal(contrast, tf.zeros_like(contrast))

        # No file_name
        assert 'file_name' not in inputs


if __name__ == '__main__':
    unittest.main()
