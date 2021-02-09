import logging
from functools import partial

import tensorflow as tf
import tensorflow_datasets as tfds

from data import augmentations
from data.preprocess import process_encoded_example


def get_val_split_name(ds_info):
    for split_name in ['validation', 'test', 'train']:
        if split_name in ds_info.splits:
            return split_name


def add_batch_sims(inputs):
    labels = inputs['label']
    labels = tf.expand_dims(labels, axis=1)
    tf.debugging.assert_shapes([
        (labels, [None, 1])
    ])
    class_sims = tf.cast(labels == tf.transpose(labels), tf.uint8)
    contrast = class_sims + tf.eye(tf.shape(labels)[0], dtype=tf.uint8)
    tf.debugging.assert_shapes([
        (contrast, ('N', 'N'))
    ])
    inputs['contrast'] = contrast
    return inputs


def source_dataset(input_ctx, ds_info, data_id, split, cache, shuffle, repeat, augment_config, global_bsz):
    # Load image bytes and labels
    decoder_args = {'image': tfds.decode.SkipDecoding()}
    read_config = tfds.ReadConfig(input_context=input_ctx)
    ds = tfds.load(data_id, read_config=read_config, split=split, shuffle_files=shuffle, decoders=decoder_args,
                   try_gcs=True, data_dir='gs://aigagror/datasets')
    imsize = ds_info.features['image'].shape[0] or 224

    # Cache?
    if cache:
        ds = ds.cache()
        logging.info(f'caching {split} dataset')

    # Shuffle?
    if shuffle:
        ds = ds.shuffle(10000)
        logging.info(f'shuffling {split} dataset with buffer size 10000')

    # Repeat infinitely?
    if repeat:
        ds = ds.repeat()
        logging.info(f'repeat {split} dataset')

    # Preprocess
    preprocess_fn = partial(process_encoded_example, imsize=imsize, augment_config=augment_config)
    ds = ds.map(preprocess_fn, tf.data.AUTOTUNE)

    # Batch
    per_replica_bsz = input_ctx.get_per_replica_batch_size(global_bsz)
    ds = ds.batch(per_replica_bsz)

    # Add batch similarities (supcon labels)
    if len(augment_config.view_configs) > 1:
        ds = ds.map(add_batch_sims, tf.data.AUTOTUNE)
        logging.info('added batch similarities')

    # Prefetch
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


def load_distributed_datasets(args, ds_info, strategy, train_augment_config, val_augment_config):
    ds_train_fn = partial(source_dataset, ds_info=ds_info, data_id=args.data_id, split='train', cache=args.cache,
                          shuffle=True, repeat=True, augment_config=train_augment_config, global_bsz=args.bsz)
    val_split_name = get_val_split_name(ds_info)
    ds_val_fn = partial(source_dataset, ds_info=ds_info, data_id=args.data_id, split=val_split_name, cache=False,
                        shuffle=False, repeat=False, augment_config=val_augment_config, global_bsz=args.bsz)

    ds_train = strategy.distribute_datasets_from_function(ds_train_fn)
    ds_val = strategy.distribute_datasets_from_function(ds_val_fn)
    return ds_train, ds_val
