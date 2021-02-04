import unittest

import matplotlib.pyplot as plt
import tensorflow as tf

import data
import data.cifar10
import data.imagenet
import logging
import data.preprocess
import utils


class TestData(unittest.TestCase):

    def test_preprocess_for_train_and_eval(self):
        image_bytes = tf.io.read_file('images/imagenet-sample.jpg')

        for name, preprocess_fn in [('preprocess-for-train', data.preprocess.preprocess_for_train),
                                    ('preprocess-for-eval', data.preprocess.preprocess_for_eval)]:
            f, ax = plt.subplots(1, 9)
            f.set_size_inches(20, 5)
            ax[0].set_title('original')
            ax[0].imshow(tf.image.decode_image(image_bytes))
            for i in range(1, 9):
                new_img = preprocess_fn(image_bytes, 224)
                tf.debugging.assert_type(new_img, tf.uint8)
                ax[i].set_title('augmentation')
                ax[i].imshow(new_img)
            f.tight_layout()
            f.savefig(f'images/{name}.jpg')
        logging.info("test images saved to 'data/images/'")

    def test_data_format(self):
        args = '--data=cifar10 --bsz=8 --loss=supcon'
        args = utils.parser.parse_args(args.split())
        _ = utils.setup(args)
        ds_train, ds_val, ds_info = data.load_datasets(args)

        for ds in [ds_train, ds_val]:
            inputs, targets = next(iter(ds))
            img = inputs['imgs']

            # Image
            tf.debugging.assert_shapes([
                (img, [8, 32, 32, 3])
            ])
            tf.debugging.assert_type(img, tf.uint8, img.dtype)
            tf.debugging.assert_greater_equal(img, tf.zeros_like(img))
            tf.debugging.assert_less_equal(img, 255 * tf.ones_like(img))
            max_val = tf.reduce_max(img)
            tf.debugging.assert_greater(max_val, tf.ones_like(max_val), 'although not necessary. it is highly unlikely '
                                                                        'that the largest pixel value of an image is at '
                                                                        'most 1.')

            # Label
            label = targets['labels']
            tf.debugging.assert_shapes([(label, [8])])
            tf.debugging.assert_type(label, tf.int32, label.dtype)
            tf.debugging.assert_less_equal(label, ds_info['nclass'] - 1, label)
            tf.debugging.assert_greater_equal(label, 0)

            # Contrast
            contrast = targets['contrast']
            tf.debugging.assert_shapes([(contrast, [8, 8])])
            tf.debugging.assert_type(contrast, tf.uint8, contrast.dtype)
            tf.debugging.assert_less_equal(contrast, 2 * tf.ones_like(contrast))
            tf.debugging.assert_greater_equal(contrast, tf.zeros_like(contrast))

    def test_autoaugment(self):
        self.skipTest('too long')
        all_args = ['--data=cifar10 --bsz=8 --shuffle-buffer=0',
                    '--data=cifar10 --autoaugment --bsz=8 --shuffle-buffer=0']
        for args in all_args:
            args = utils.parser.parse_args(args.split())

            ds_train, ds_val, ds_info = data.load_datasets(args)
            imgs = next(iter(ds_train))[0]['imgs']
            f, ax = plt.subplots(1, 8)
            f.set_size_inches(20, 5)
            for i in range(8):
                ax[i].imshow(imgs[i])
            f.tight_layout()
            f.savefig(f'images/cifar10-autoaugment-{args.autoaugment}.jpg')

        logging.info("test images saved to 'data/images/'")


if __name__ == '__main__':
    unittest.main()
