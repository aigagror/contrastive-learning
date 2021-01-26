import unittest

import matplotlib.pyplot as plt
import tensorflow as tf

import data
import data.cifar10
import data.data_utils
import data.imagenet
import utils


class TestData(unittest.TestCase):

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

    def test_imagenet_augmentation(self):
        img = tf.io.decode_image(tf.io.read_file('images/imagenet-sample.jpg'))

        f, ax = plt.subplots(1, 9)
        f.set_size_inches(20, 5)
        ax[0].set_title('original')
        ax[0].imshow(img)
        for i in range(1, 9):
            aug_img = data.imagenet.augment_imagenet_img(img)
            tf.debugging.assert_type(aug_img, tf.uint8)
            ax[i].set_title('augmentation')
            ax[i].imshow(aug_img)
        f.tight_layout()
        f.savefig('images/imagenet-sample-augmentations.jpg')
        print("Test images saved to 'data/images/'")

    def test_cifar10_augmentation(self):
        img = tf.io.decode_image(tf.io.read_file('images/cifar10-sample.png'))

        f, ax = plt.subplots(1, 9)
        f.set_size_inches(20, 5)
        ax[0].set_title('original')
        ax[0].imshow(img)
        for i in range(1, 9):
            aug_img = data.cifar10.augment_cifar10_img(img)
            tf.debugging.assert_type(aug_img, tf.uint8)
            ax[i].set_title('augmentation')
            ax[i].imshow(aug_img)
        f.tight_layout()
        f.savefig('images/cifar10-sample-augmentations.jpg')
        print("Test images saved to 'data/images/'")

    def test_min_scale_crops(self):
        img = tf.io.decode_image(tf.io.read_file('images/imagenet-sample.jpg'), channels=3)
        tf.debugging.assert_shapes([(img, [None, None, 3])])

        for resize_fn in [data.data_utils.min_scale_rand_crop, data.data_utils.min_scale_center_crop]:
            f, ax = plt.subplots(1, 9)
            f.set_size_inches(20, 5)
            ax[0].set_title('original')
            ax[0].imshow(img)

            for i in range(1, 9):
                resized_img = resize_fn(img, 224)
                tf.debugging.assert_type(resized_img, tf.uint8)
                ax[i].set_title('resize')
                ax[i].imshow(resized_img)
            f.tight_layout()
            f.savefig(f'images/test-{resize_fn.__name__}.jpg')
        print("Test images saved to 'data/images/'")


if __name__ == '__main__':
    unittest.main()
