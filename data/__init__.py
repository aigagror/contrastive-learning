import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import datasets
from tensorflow.python.data import AUTOTUNE

from data import serial


def augment(image):
    # Convert to float
    image = tf.image.convert_image_dtype(image, tf.float32)
    imsize = image.shape[0]

    # Crop
    rand_scale = tf.random.uniform([], 1, 2)
    rand_size = tf.round(rand_scale * imsize)
    image = tf.image.resize(image, [rand_size, rand_size])
    image = tf.image.random_crop(image, [imsize, imsize, 3])

    # Random flip
    image = tf.image.random_flip_left_right(image)

    # Color Jitter
    if tf.random.uniform([]) < 0.8:
        image = tf.image.random_brightness(image, 0.4)
        image = tf.image.random_contrast(image, 0.6, 1.4)
        image = tf.image.random_saturation(image, 0.6, 1.4)
        image = tf.image.random_hue(image, 0.1)

    # Gray scale
    if tf.random.uniform([]) < 0.2:
        image = tf.image.rgb_to_grayscale(image)
        image = tf.tile(image, [1, 1, 3])

    # Clip
    image = tf.clip_by_value(image, 0, 1)

    return image


def load_datasets(args, strategy):
    if args.data == 'cifar10':
        imsize = 32
        (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
        ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train.flatten())).cache()
        ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test.flatten())).cache()

    elif args.data == 'imagenet':
        imsize = 224
        ds_train = tfds.folder_dataset.ImageFolder(args.imagenet_train).as_dataset(shuffle_files=True)
        ds_test = tfds.folder_dataset.ImageFolder(args.imagenet_val).as_dataset(shuffle_files=True)
    else:
        raise Exception(f'unknown data {args.data}')

    # Map functions
    def resize(img, labels):
        return tf.image.resize(img, [imsize, imsize]), labels

    def dual_augment(imgs, labels):
        return augment(imgs), augment(imgs), labels

    def dual_views(imgs, labels):
        imgs = tf.image.convert_image_dtype(imgs, tf.float32)
        return imgs, imgs, labels

    # Preprocess
    ds_train = (
        ds_train
            .map(resize, num_parallel_calls=AUTOTUNE)
            .map(dual_augment, num_parallel_calls=AUTOTUNE)
            .shuffle(1024)
            .batch(args.bsz, drop_remainder=True)
            .prefetch(AUTOTUNE)
    )
    ds_test = (
        ds_test
            .map(resize, num_parallel_calls=AUTOTUNE)
            .map(dual_views, num_parallel_calls=AUTOTUNE)
            .shuffle(1024)
            .batch(args.bsz, drop_remainder=True)
            .prefetch(AUTOTUNE)
    )

    ds_train = strategy.experimental_distribute_dataset(ds_train)
    ds_test = strategy.experimental_distribute_dataset(ds_test)

    return ds_train, ds_test
