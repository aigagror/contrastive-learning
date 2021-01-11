import unittest

import matplotlib.pyplot as plt
import tensorflow as tf

import data
import utils


class TestData(unittest.TestCase):

    def test_image_label_range(self):
        args = '--data=cifar10 --bsz=32 --method=ce'
        args = utils.parser.parse_args(args.split())
        _ = utils.setup(args)
        ds_train, ds_val, nclass = data.load_datasets(args)

        for ds in [ds_train, ds_val]:
            input = next(iter(ds))
            img, label = input['imgs'], input['labels']

            # Label
            tf.debugging.assert_shapes([(label, [None])])
            label = tf.cast(label, tf.int32)
            tf.debugging.assert_less_equal(label, nclass - 1, label)
            tf.debugging.assert_greater_equal(label, 0)

            # Image
            tf.debugging.assert_type(img, tf.float32, f'image was of type {img.dtype}.')
            tf.debugging.assert_greater_equal(img, tf.zeros_like(img))
            tf.debugging.assert_less_equal(img, 255 * tf.ones_like(img))
            tf.debugging.assert_greater(tf.reduce_max(img), 1.0, 'although not necessary. it is highly unlikely '
                                                                 'that the largest pixel value of an image is at '
                                                                 'most 1.')

    def test_augmentation(self):
        img = tf.io.decode_image(tf.io.read_file('images/imagenet-sample.jpg'))

        f, ax = plt.subplots(1, 9)
        f.set_size_inches(20, 5)
        ax[0].set_title('original')
        ax[0].imshow(tf.cast(img, tf.uint8))
        for i in range(1, 9):
            aug_img = data.augment_img(img)
            ax[i].set_title('augmentation')
            ax[i].imshow(tf.cast(aug_img, tf.uint8))
        f.tight_layout()
        f.savefig('images/test-augmentation.jpg')

    def test_resize(self):
        img = tf.io.decode_image(tf.io.read_file('images/imagenet-sample.jpg'), channels=3)
        tf.debugging.assert_shapes([(img, [None, None, 3])])

        f, ax = plt.subplots(1, 9)
        f.set_size_inches(20, 5)
        ax[0].set_title('original')
        ax[0].imshow(tf.cast(img, tf.uint8))

        for i in range(1, 9):
            aug_img = data.resize(img, 224, crop='rand')
            ax[i].set_title('resize')
            ax[i].imshow(tf.cast(aug_img, tf.uint8))
        f.tight_layout()
        f.savefig('images/test-resize.jpg')


if __name__ == '__main__':
    unittest.main()
