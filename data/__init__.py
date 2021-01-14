from tensorflow.python.data import AUTOTUNE

from data.cifar10 import load_cifar10
from data.imagenet import load_imagenet


def load_datasets(args):
    if args.data == 'cifar10':
        ds_train, ds_val, info = load_cifar10(args)

    elif args.data == 'imagenet':
        ds_train, ds_val, info = load_imagenet(args)

    else:
        raise Exception(f'unknown data {args.data}')

    # Batch and prefetch
    ds_train = ds_train.batch(args.bsz, drop_remainder=True).prefetch(AUTOTUNE)
    ds_val = ds_val.batch(args.bsz, drop_remainder=True).prefetch(AUTOTUNE)

    return ds_train, ds_val, info
