import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, datasets
from tensorflow.python.data import AUTOTUNE


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


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


feature_description = {
    'image': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
    'label': tf.io.FixedLenFeature([], tf.int64, default_value=0),
}


def serialize_example(img, label):
    """
    Creates a tf.train.Example message ready to be written to a file.
    """
    # Create a dictionary mapping the feature name to the tf.train.Example-compatible
    # data type.
    feature = {
        'image': _int64_feature(img),
        'label': _int64_feature(label)
    }

    # Create a Features message using tf.train.Example.

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def tf_serialize_example(img, label):
    tf_string = tf.py_function(
        serialize_example,
        (img, label),  # pass these args to the above function.
        tf.string)  # the return type is `tf.string`.
    return tf.reshape(tf_string, ())  # The result is a scalar


def load_datasets(args, strategy):
    if args.data == 'cifar10':
        imsize = 32
        (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
        ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train.flatten()))
        ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test.flatten()))

        serialized_ds_train = ds_train.map(tf_serialize_example)

        filename = 'cifar10-train.tfrecord'
        writer = tf.data.experimental.TFRecordWriter(filename)
        writer.write(serialized_ds_train)

        raw_dataset = tf.data.TFRecordDataset([filename])
        def _parse_function(example_proto):
            # Parse the input `tf.train.Example` proto using the dictionary above.
            example = tf.io.parse_single_example(example_proto, feature_description)
            print(example)
            return example['image'], example['label']

        ds_train = raw_dataset.map(_parse_function)

    elif args.data == 'imagenet':
        imsize = 224
        ds_train = tfds.folder_dataset.ImageFolder(args.imagenet_train).as_dataset(shuffle_files=True)
        ds_test = tfds.folder_dataset.ImageFolder(args.imagenet_val).as_dataset(shuffle_files=True)
    else:
        raise Exception(f'unknown data {args.data}')

    augment = Augment(imsize, rand_crop=True, rand_flip=True, rand_jitter=True, rand_gray=True)

    def dual_augment(imgs, labels):
        return augment(imgs), augment(imgs), labels

    def dual_views(imgs, labels):
        imgs = tf.image.convert_image_dtype(imgs, tf.float32)
        return imgs, imgs, labels

    ds_train = (
        ds_train
            .map(dual_augment, num_parallel_calls=AUTOTUNE)
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
