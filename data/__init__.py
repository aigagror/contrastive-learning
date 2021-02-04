import logging

import tensorflow as tf

from data import autoaugment
from data.cifar10 import load_cifar10
from data.imagenet import load_imagenet


def add_contrast_data(inputs, targets):
    labels = targets['labels']
    labels = tf.expand_dims(labels, axis=1)
    tf.debugging.assert_shapes([
        (labels, [None, 1])
    ])
    class_sims = tf.cast(labels == tf.transpose(labels), tf.uint8)
    contrast = class_sims + tf.eye(tf.shape(labels)[0], dtype=tf.uint8)
    tf.debugging.assert_shapes([
        (contrast, ('N', 'N'))
    ])
    targets['contrast'] = contrast
    return inputs, targets


def autoaugment_all_views(inputs, targets):
    for key in ['imgs', 'imgs2']:
        if key in inputs:
            inputs[key] = autoaugment.AutoAugment().distort(inputs[key])
    return inputs, targets


def autoaugment_second_view(inputs, targets):
    if 'imgs2' in inputs:
        inputs['imgs2'] = autoaugment.AutoAugment().distort(inputs['imgs2'])
    return inputs, targets


def load_datasets(args):
    if 'cifar10' in args.data:
        ds_train, ds_val, ds_info = load_cifar10(args)
        if args.data.startswith('fake-'):
            ds_train = ds_train.take(4)
            ds_val = ds_val.take(4)
    elif args.data == 'imagenet':
        ds_train, ds_val, ds_info = load_imagenet(args)
    else:
        raise Exception(f'unknown data {args.data}')

    # Shuffle?
    shuffle = args.shuffle_buffer is not None and args.shuffle_buffer > 0
    if shuffle:
        ds_train = ds_train.shuffle(args.shuffle_buffer)
        logging.info('shuffling dataset')

    # Repeat train dataset
    ds_train = ds_train.repeat()

    # Autoaugment
    if args.autoaugment:
        ds_train = ds_train.map(autoaugment_all_views, tf.data.AUTOTUNE)
        ds_val = ds_val.map(autoaugment_second_view, tf.data.AUTOTUNE)
        logging.info('autoaugment-ed datasets')

    # Batch
    ds_train = ds_train.batch(args.bsz)
    ds_val = ds_val.batch(args.bsz)

    # Add batch similarities (supcon labels)
    if args.loss != 'ce':
        ds_train = ds_train.map(add_contrast_data, tf.data.AUTOTUNE)
        ds_val = ds_val.map(add_contrast_data, tf.data.AUTOTUNE)
        logging.info('addded contrast data')

    # Prefetch
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)
    ds_val = ds_val.prefetch(tf.data.AUTOTUNE)

    return ds_train, ds_val, ds_info
