import tensorflow as tf

from data.cifar10 import load_cifar10
from data.imagenet import load_imagenet


def add_batch_sims(inputs, targets):
    labels = targets['labels']
    labels = tf.expand_dims(labels, axis=1)
    tf.debugging.assert_shapes([
        (labels, [None, 1])
    ])
    class_sims = tf.cast(labels == tf.transpose(labels), tf.uint8)
    targets['batch_sims'] = class_sims + tf.eye(tf.shape(labels)[0], dtype=tf.uint8)
    return inputs, targets


def load_datasets(args):
    if args.data == 'cifar10':
        ds_train, ds_val, ds_info = load_cifar10(args)

    elif args.data == 'imagenet':
        ds_train, ds_val, ds_info = load_imagenet(args)

    else:
        raise Exception(f'unknown data {args.data}')

    # Batch
    ds_train = ds_train.batch(args.bsz, drop_remainder=True)
    ds_val = ds_val.batch(args.bsz, drop_remainder=True)

    # Add batch similarities (supcon labels)
    if args.method != 'ce':
        ds_train = ds_train.map(add_batch_sims, tf.data.AUTOTUNE)
        ds_val = ds_val.map(add_batch_sims, tf.data.AUTOTUNE)

    # Prefetch
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)
    ds_val = ds_val.prefetch(tf.data.AUTOTUNE)

    return ds_train, ds_val, ds_info
