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
            tf.debugging.assert_type(img, tf.uint8, f'image was of type {img.dtype}.')
            tf.debugging.assert_greater_equal(img, tf.zeros_like(img))
            tf.debugging.assert_less_equal(img, 255 * tf.ones_like(img))
            max_val = tf.reduce_max(img)
            tf.debugging.assert_greater(max_val, tf.ones_like(max_val), 'although not necessary. it is highly unlikely '
                                                                        'that the largest pixel value of an image is at '
                                                                        'most 1.')

    def test_augmentation(self):
        img = tf.io.decode_image(tf.io.read_file('images/imagenet-sample.jpg'))

        f, ax = plt.subplots(1, 9)
        f.set_size_inches(20, 5)
        ax[0].set_title('original')
        ax[0].imshow(img)
        for i in range(1, 9):
            aug_img = data.augment_img(img)
            ax[i].set_title('augmentation')
            ax[i].imshow(aug_img)
        f.tight_layout()
        f.savefig('images/test-augmentation.jpg')

    def test_resize(self):
        img = tf.io.decode_image(tf.io.read_file('images/imagenet-sample.jpg'), channels=3)
        tf.debugging.assert_shapes([(img, [None, None, 3])])

        for resize_fn in [data.rand_resize, data.center_resize]:
            f, ax = plt.subplots(1, 9)
            f.set_size_inches(20, 5)
            ax[0].set_title('original')
            ax[0].imshow(img)

            for i in range(1, 9):
                resized_img = resize_fn(img, 224)
                ax[i].set_title('resize')
                ax[i].imshow(resized_img)
            f.tight_layout()
            f.savefig(f'images/test-{resize_fn.__name__}.jpg')


if __name__ == '__main__':
    unittest.main()
