import tensorflow as tf
from tensorflow.python.data import AUTOTUNE

from data import autoaugment
from data.preprocess import preprocess_for_train, preprocess_for_eval


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
    label = example['image/class/label'] - 1
    return example['image/encoded'], label


def load_imagenet(args):
    imsize, nclass = 224, 1000
    train_size, val_size = 1281167, 50000
    train_files = tf.data.Dataset.list_files('gs://aigagror/datasets/imagenet/train*', shuffle=True)
    val_files = tf.data.Dataset.list_files('gs://aigagror/datasets/imagenet/validation-*', shuffle=True)
    train_data = train_files.interleave(tf.data.TFRecordDataset, cycle_length=10, num_parallel_calls=AUTOTUNE)
    val_data = val_files.interleave(tf.data.TFRecordDataset, num_parallel_calls=AUTOTUNE)

    # Shuffle and repeat train data
    train_data = train_data.shuffle(10000)
    train_data = train_data.repeat()

    ds_train = train_data.map(parse_imagenet_example, AUTOTUNE)
    ds_val = val_data.map(parse_imagenet_example, AUTOTUNE)

    # Get augment
    augment = None
    if args.augment == 'auto':
        augment = autoaugment.AutoAugment().distort

    # Preprocess
    if args.loss == 'ce':
        def process_train(img_bytes, label):
            inputs = {'imgs': preprocess_for_train(img_bytes, 224, augment)}
            targets = {'labels': label}
            return inputs, targets

        def process_val(img_bytes, label):
            inputs = {'imgs': preprocess_for_eval(img_bytes, 224)}
            targets = {'labels': label}
            return inputs, targets
    else:
        def process_train(img_bytes, label):
            inputs = {'imgs': preprocess_for_train(img_bytes, 224, augment),
                      'imgs2': preprocess_for_train(img_bytes, 224, augment)}
            targets = {'labels': label}
            return inputs, targets

        def process_val(img_bytes, label):
            inputs = {'imgs': preprocess_for_eval(img_bytes, 224),
                      'imgs2': preprocess_for_train(img_bytes, 224, augment)}
            targets = {'labels': label}
            return inputs, targets

    ds_train = ds_train.map(process_train, AUTOTUNE)
    ds_val = ds_val.map(process_val, AUTOTUNE)
    info = {'nclass': nclass, 'input_shape': [imsize, imsize, 3], 'train_size': train_size, 'val_size': val_size}
    return ds_train, ds_val, info
