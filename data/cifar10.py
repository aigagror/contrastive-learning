import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.python.data import AUTOTUNE

from data.data_utils import color_augment


def augment_cifar10_img(image):
    # Pad 4 pixels on all sides
    image = tf.image.pad_to_bounding_box(image, 4, 4, 40, 40)

    # Random crop
    image = tf.image.random_crop(image, [32, 32, 3])

    # Random flip
    image = tf.image.random_flip_left_right(image)

    # Color augment
    image = color_augment(image)

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
    if args.loss == 'ce':
        def process_train(img, label):
            inputs = {'imgs': augment_cifar10_img(img)}
            targets = {'labels': label}
            return inputs, targets

        def process_val(img, label):
            return {'imgs': img}, {'labels': label}
    else:
        def process_train(img, label):
            inputs = {'imgs': augment_cifar10_img(img), 'imgs2': augment_cifar10_img(img)}
            targets = {'labels': label}
            return inputs, targets

        def process_val(img, label):
            return {'imgs': img, 'imgs2': augment_cifar10_img(img)}, {'labels': label}

    ds_train = ds_train.map(process_train, AUTOTUNE)
    ds_val = ds_val.map(process_val, AUTOTUNE)
    info = {'nclass': nclass, 'input_shape': [imsize, imsize, 3],
            'train_size': 50000, 'val_size': 10000}
    return ds_train, ds_val, info
