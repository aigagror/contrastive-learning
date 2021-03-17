from functools import partial

import tensorflow as tf
import tensorflow_datasets as tfds
from absl import logging

from data import augmentations
from data.preprocess import process_encoded_example


def get_val_split_name(ds_info):
    for split_name in ['validation', 'test', 'train']:
        if split_name in ds_info.splits:
            return split_name


def add_contrast_data(inputs, targets):
    targets['contrast'] = targets['label']
    return inputs, targets


def source_dataset(input_ctx, ds_info, data_id, split, cache, shuffle, repeat, augment_config, global_bsz):
    # Load image bytes and labels
    decoder_args = {'image': tfds.decode.SkipDecoding()}
    read_config = tfds.ReadConfig(input_context=input_ctx)
    ds = tfds.load(data_id, as_supervised=True, read_config=read_config, split=split, shuffle_files=shuffle,
                   decoders=decoder_args, try_gcs=True, data_dir='gs://aigagror/datasets')
    imsize = ds_info.features['image'].shape[0] or 224
    channels = ds_info.features['image'].shape[2]

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
    preprocess_fn = partial(process_encoded_example, imsize=imsize, channels=channels, augment_config=augment_config)
    ds = ds.map(preprocess_fn, tf.data.AUTOTUNE)

    # Batch
    per_replica_bsz = input_ctx.get_per_replica_batch_size(global_bsz)
    ds = ds.batch(per_replica_bsz, drop_remainder=True)

    if len(augment_config.view_configs) > 1:
        ds = ds.map(add_contrast_data, tf.data.AUTOTUNE)

    # Prefetch
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


def load_distributed_datasets(args, strategy, ds_info, split, train_augment_config):
    ds_fn = partial(source_dataset, ds_info=ds_info, data_id=args.data_id, split=split, cache=args.cache,
                    shuffle=True, repeat=True, augment_config=train_augment_config, global_bsz=args.bsz)

    ds = strategy.distribute_datasets_from_function(ds_fn)
    return ds
