import tensorflow_datasets as tfds
from tensorflow.python.data import AUTOTUNE

from data.preprocess import preprocess_for_train, preprocess_for_eval


def load_imagenet(args):
    imsize, nclass = 224, 1000
    train_size, val_size = 1281167, 50000

    # Shuffle?
    shuffle = args.shuffle_buffer is not None and args.shuffle_buffer > 0

    # Sharded record datasets
    decoder_args = {'image': tfds.decode.SkipDecoding()}
    ds_train, info = tfds.load('imagenet2012', split='train', shuffle_files=shuffle, with_info=True, as_supervised=True,
                               decoders=decoder_args, data_dir='gs://aigagror/datasets')
    ds_val = tfds.load('imagenet2012', split='validation', as_supervised=True, decoders=decoder_args,
                       data_dir='gs://aigagror/datasets')

    # Preprocess
    if args.loss == 'ce':
        def process_train(img_bytes, label):
            inputs = {'imgs': preprocess_for_train(img_bytes, imsize)}
            targets = {'labels': label}
            return inputs, targets

        def process_val(img_bytes, label):
            inputs = {'imgs': preprocess_for_eval(img_bytes, imsize)}
            targets = {'labels': label}
            return inputs, targets
    else:
        def process_train(img_bytes, label):
            inputs = {'imgs': preprocess_for_train(img_bytes, imsize),
                      'imgs2': preprocess_for_train(img_bytes, imsize)}
            targets = {'labels': label}
            return inputs, targets

        def process_val(img_bytes, label):
            inputs = {'imgs': preprocess_for_eval(img_bytes, imsize),
                      'imgs2': preprocess_for_train(img_bytes, imsize)}
            targets = {'labels': label}
            return inputs, targets

    ds_train = ds_train.map(process_train, AUTOTUNE)
    ds_val = ds_val.map(process_val, AUTOTUNE)
    info = {'nclass': nclass, 'input_shape': [imsize, imsize, 3], 'train_size': train_size, 'val_size': val_size}
    return ds_train, ds_val, info
