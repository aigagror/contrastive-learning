import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.python.data import AUTOTUNE


def parse_imagenet_example(serial):
    features = {
        'image/height': tf.io.FixedLenFeature([], tf.int64),
        'image/width': tf.io.FixedLenFeature([], tf.int64),
        'image/colorspace': tf.io.FixedLenFeature([], tf.string),
        'image/channels': tf.io.FixedLenFeature([], tf.int64),
        'image/class/label': tf.io.FixedLenFeature([], tf.int64),
        'image/class/synset': tf.io.FixedLenFeature([], tf.string),
        'image/format': tf.io.FixedLenFeature([], tf.string),
        'image/filename': tf.io.FixedLenFeature([], tf.string),
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_example(serial, features)
    img = tf.io.decode_image(example['image/encoded'], channels=3, expand_animations=False)
    label = example['image/class/label'] - 1
    return img, label


def scale_min_dim(img, imsize):
    imshape = tf.cast(tf.shape(img), tf.float32)
    h, w = imshape[0], imshape[1]
    small_length = tf.minimum(h, w)
    scale = tf.cast(imsize + 1, tf.float32) / small_length
    new_size = [tf.cast(h * scale, tf.int32), tf.cast(w * scale, tf.int32)]
    img = tf.image.resize(img, new_size)
    return img


def min_scale_rand_crop(img, imsize):
    img = scale_min_dim(img, imsize)
    img = tf.image.random_crop(img, [imsize, imsize, 3])
    img = tf.cast(img, tf.uint8)
    return img


def min_scale_center_crop(img, imsize):
    img = scale_min_dim(img, imsize)
    img = tf.image.resize_with_crop_or_pad(img, imsize, imsize)
    img = tf.cast(img, tf.uint8)
    return img


def augment_imagenet_img(image):
    """
    From original resnet paper
    https://arxiv.org/pdf/1512.03385.pdf
    :param image:
    :return:
    """

    # Random scale
    rand_size = tf.random.uniform([], 256, 481, tf.int32)
    image = scale_min_dim(image, rand_size)

    # Random crop
    image = tf.image.random_crop(image, [224, 224, 3])

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
        image = tf.repeat(image, 3, axis=-1)

    # Clip
    image = tf.clip_by_value(image, 0, 255)
    image = tf.cast(image, tf.uint8)
    return image


def load_imagenet(args):
    imsize, nclass = 224, 1000
    train_files = tf.data.Dataset.list_files('gs://aigagror/datasets/imagenet/train*', shuffle=True)
    val_files = tf.data.Dataset.list_files('gs://aigagror/datasets/imagenet/validation-*', shuffle=True)
    train_data = train_files.interleave(tf.data.TFRecordDataset, num_parallel_calls=AUTOTUNE)
    val_data = val_files.interleave(tf.data.TFRecordDataset, num_parallel_calls=AUTOTUNE)
    ds_train = train_data.map(parse_imagenet_example, AUTOTUNE)
    ds_val = val_data.map(parse_imagenet_example, AUTOTUNE)

    # Preprocess
    def process_train(img, label):
        ret = {'imgs': augment_imagenet_img(img), 'labels': label}
        if args.method.startswith('supcon'):
            ret['imgs2'] = augment_imagenet_img(img)
        return ret

    def process_val(img, label):
        return {'imgs': min_scale_rand_crop(img, 224), 'imgs2': augment_imagenet_img(img), 'labels': label}

    ds_train = ds_train.map(process_train, AUTOTUNE)
    ds_val = ds_val.map(process_val, AUTOTUNE)
    return ds_train, ds_val, nclass


def augment_cifar10_img(image):
    # Pad 4 pixels on all sides
    image = tf.image.pad_to_bounding_box(image, 4, 4, 40, 40)

    # Random crop
    image = tf.image.random_crop(image, [32, 32, 3])

    # Random flip
    image = tf.image.random_flip_left_right(image)

    image = tf.cast(image, tf.uint8)
    return image


def load_cifar10(args):
    imsize, nclass = 32, 10
    (x_train, y_train), (x_val, y_val) = datasets.cifar10.load_data()
    y_train = y_train.astype(np.int32)
    y_val = y_val.astype(np.int32)

    # Cache because it's small
    ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train.flatten())).cache()
    ds_val = tf.data.Dataset.from_tensor_slices((x_val, y_val.flatten())).cache()

    # Shuffle entire dataset
    ds_train = ds_train.shuffle(len(ds_train))
    ds_val = ds_val.shuffle(len(ds_val))

    # Preprocess
    def process_train(img, label):
        ret = {'imgs': augment_cifar10_img(img), 'labels': label}
        if args.method.startswith('supcon'):
            ret['imgs2'] = augment_cifar10_img(img)
        return ret

    def process_val(img, label):
        return {'imgs': img, 'imgs2': augment_cifar10_img(img), 'labels': label}

    ds_train = ds_train.map(process_train, AUTOTUNE)
    ds_val = ds_val.map(process_val, AUTOTUNE)
    return ds_train, ds_val, nclass


def load_datasets(args):
    if args.data == 'cifar10':
        ds_train, ds_val, nclass = load_cifar10(args)

    elif args.data == 'imagenet':
        ds_train, ds_val, nclass = load_imagenet(args)

    else:
        raise Exception(f'unknown data {args.data}')

    # Batch and prefetch
    ds_train = ds_train.batch(args.bsz, drop_remainder=True).prefetch(AUTOTUNE)
    ds_val = ds_val.batch(args.bsz, drop_remainder=True).prefetch(AUTOTUNE)

    return ds_train, ds_val, nclass
