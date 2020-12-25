import tensorflow as tf
from tensorflow.keras import layers, datasets
from tensorflow.python.data import AUTOTUNE
import tensorflow_datasets as tfds


class Augment(layers.Layer):
    def __init__(self, imsize, rand_crop, rand_flip, rand_jitter, rand_gray):
        super().__init__(name='image-augmentation')
        self.imsize = imsize
        self.rand_crop = rand_crop
        self.rand_flip = rand_flip
        self.rand_jitter = rand_jitter
        self.rand_gray = rand_gray

    @tf.function
    def call(self, image):
        # Convert to float
        image = tf.image.convert_image_dtype(image, tf.float32)

        # Crop
        if self.rand_crop:
            rand_scale = tf.random.uniform([], 1, 2)
            rand_size = tf.round(rand_scale * self.imsize)
            image = tf.image.resize(image, [rand_size, rand_size])
            image = tf.image.random_crop(image, [self.imsize, self.imsize, 3])
        else:
            image = tf.image.resize(image, [self.imsize, self.imsize])

        # Random flip
        if self.rand_flip:
            image = tf.image.random_flip_left_right(image)

        # Color Jitter
        if self.rand_jitter and tf.random.uniform([]) < 0.8:
            image = tf.image.random_brightness(image, 0.4)
            image = tf.image.random_contrast(image, 0.6, 1.4)
            image = tf.image.random_saturation(image, 0.6, 1.4)
            image = tf.image.random_hue(image, 0.1)

        # Gray scale
        if self.rand_gray and tf.random.uniform([]) < 0.2:
            image = tf.image.rgb_to_grayscale(image)
            image = tf.tile(image, [1, 1, 3])

        # Clip
        image = tf.clip_by_value(image, 0, 1)

        return image


def load_datasets(args, strategy):
    if args.data == 'cifar10':
        imsize = 32
        (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
        ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train.flatten()))
        ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test.flatten()))
    elif args.data =='imagenet':
        imsize = 224
        ds_train = tfds.folder_dataset.ImageFolder(args.imagenet_train).as_dataset(shuffle_files=True)
        ds_test = tfds.folder_dataset.ImageFolder(args.imagenet_val).as_dataset(shuffle_files=True)
    else:
        raise Exception(f'unknown data {args.data}')

    augment = Augment(imsize, rand_crop=True, rand_flip=True, rand_jitter=True, rand_gray=True)

    def dual_views(imgs, labels):
        return augment(imgs), augment(imgs), labels

    ds_train = (
        ds_train
            .map(dual_views, num_parallel_calls=AUTOTUNE)
            .shuffle(len(ds_train))
            .batch(args.bsz, drop_remainder=True)
            .prefetch(AUTOTUNE)
    )
    ds_test = (
        ds_test
            .map(dual_views, num_parallel_calls=AUTOTUNE)
            .shuffle(len(ds_test))
            .batch(args.bsz)
            .prefetch(AUTOTUNE)
    )

    ds_train = strategy.experimental_distribute_dataset(ds_train)
    ds_test = strategy.experimental_distribute_dataset(ds_test)

    return ds_train, ds_test
