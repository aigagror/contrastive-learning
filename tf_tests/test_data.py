import unittest

import tensorflow as tf
import tensorflow_datasets as tfds
from absl import logging

import data
import data.preprocess
import utils


class TestData(unittest.TestCase):

    def test_augmentation(self):
        data_id = 'tf_flowers'
        args = f'--data-id={data_id} --autoaugment --bsz=8 --loss=supcon '
        args = utils.parser.parse_args(args.split())
        strategy = utils.setup(args)

        _, ds_info = tfds.load(args.data_id, try_gcs=True, data_dir='gs://aigagror/datasets', with_info=True)
        train_augment_config, val_augment_config = utils.load_augment_configs(args)
        ds_train, ds_val = data.load_distributed_datasets(args, strategy, ds_info, train_augment_config,
                                                          val_augment_config)

        ds_train, ds_val = ds_train.map(lambda x, y: {**x, **y}), ds_val.map(lambda x, y: {**x, **y})
        train_fig = tfds.show_examples(ds_train.unbatch(), ds_info, rows=1)
        val_fig = tfds.show_examples(ds_val.unbatch(), ds_info, rows=1)
        train_fig.savefig(f'out/{data_id}_train_examples.jpg'), val_fig.savefig(f'out/{data_id}_val_examples.jpg')
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
        for key in ['label', 'contrast']:
            target_val = targets[key]
            tf.debugging.assert_shapes([(target_val, [8])])
            tf.debugging.assert_type(target_val, tf.int64, target_val.dtype)
            tf.debugging.assert_less_equal(target_val,
                                           tf.cast(ds_info.features['label'].num_classes - 1, target_val.dtype),
                                           target_val)
            tf.debugging.assert_greater_equal(target_val, tf.zeros_like(target_val))

        # No file_name
        assert 'file_name' not in inputs


if __name__ == '__main__':
    unittest.main()
