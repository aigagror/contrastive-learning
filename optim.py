import numpy as np
import tensorflow as tf
from tensorflow import keras

_global_lr, _global_optimizer = None, None


def set_global_optimizer(args):
    global _global_optimizer
    _global_optimizer = keras.optimizers.SGD(get_global_lr, momentum=0.9)


def get_global_optimizer():
    global _global_optimizer
    return _global_optimizer


def ndevices(args):
    if args.tpu:
        devices = tf.config.list_logical_devices('TPU')
    else:
        devices = tf.config.list_logical_devices('GPU')
    return len(devices)


def set_global_lr(args, epoch):
    global _global_lr
    n = 32
    k = ndevices(args)
    ref_lr = 0.1 * n * k / 256
    warmup = np.linspace(0.1, ref_lr, 5)
    decays = ref_lr * 0.1 ** np.arange(1, 4)
    values = np.append(warmup, decays).tolist()
    boundaries = [0, 1, 2, 3, 29, 59, 79]
    lr_fn = keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
    _global_lr = lr_fn(epoch).numpy()
    return _global_lr


def get_global_lr():
    global _global_lr
    return _global_lr
