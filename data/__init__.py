import logging
from functools import partial

import tensorflow as tf
import tensorflow_datasets as tfds

from data import autoaugment
from data.cifar10 import load_cifar10
from data.imagenet import load_imagenet
from data.preprocess import process_encoded_example, augment


def add_batch_sims(inputs, targets):
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


def load_datasets(args, views, with_batch_sims):
    # Load image bytes and labels
    shuffle = args.shuffle_buffer is not None and args.shuffle_buffer > 0
    decoder_args = {'image': tfds.decode.SkipDecoding()}
    splits = ['train', 'validation' if args.data == 'imagenet' else 'test']
    ds_train, ds_val, info = tfds.load(args.data, split=splits, as_supervised=True, shuffle_files=shuffle,
                                       decoders=decoder_args, with_info=True, data_dir='gs://aigagror/datasets')

    # Determine image size
    imsize = {'cifar10': 32, 'cifar100': 32, 'imagenet2012': 224}[args.data]

    # Preprocess
    process_train_fn = partial(process_encoded_example, views=views, imsize=imsize, rand_crop=True)
    process_val_fn = partial(process_encoded_example, views=views, imsize=imsize, rand_crop=False)

    ds_train = ds_train.map(process_train_fn, tf.data.AUTOTUNE)
    ds_val = ds_val.map(process_val_fn, tf.data.AUTOTUNE)

    # Augment (we skip augmenting the first view in the validation set)
    if args.autoaugment:
        autoaugment_fn = autoaugment.AutoAugment().distort
        augment_fn = lambda x: autoaugment_fn(tf.image.random_flip_left_right(x))
    else:
        augment_fn = tf.image.random_flip_left_right

    augment_train_fn = partial(augment, views=views, augment_fn=augment_fn)
    augment_val_fn = partial(augment, views=views[1:], augment_fn=augment_fn)
    ds_train = ds_train.map(augment_train_fn, tf.data.AUTOTUNE)
    ds_val = ds_val.map(augment_val_fn, tf.data.AUTOTUNE)

    # Shuffle?
    if shuffle:
        ds_train = ds_train.shuffle(args.shuffle_buffer)
        logging.info(f'train dataset shuffled with {args.shuffle_buffer} buffer')

    # Repeat train dataset
    ds_train = ds_train.repeat()

    # Batch
    ds_train = ds_train.batch(args.bsz)
    ds_val = ds_val.batch(args.bsz)

    # Add batch similarities (supcon labels)
    if with_batch_sims:
        ds_train = ds_train.map(add_batch_sims, tf.data.AUTOTUNE)
        ds_val = ds_val.map(add_batch_sims, tf.data.AUTOTUNE)
        logging.info('added batch similarities')

    # Prefetch
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)
    ds_val = ds_val.prefetch(tf.data.AUTOTUNE)

    return ds_train, ds_val, info
