import argparse
import logging
import os
import shutil

import tensorflow as tf
from tensorflow.keras import mixed_precision

from data import augmentations
from models import custom_layers
from training import custom_losses, lr_schedule

parser = argparse.ArgumentParser()

# Data
parser.add_argument('--data-id', choices=['imagenet2012', 'tf_flowers', 'cifar10', 'cifar100', 'mnist'])
parser.add_argument('--autoaugment', action='store_true')
parser.add_argument('--cache', action='store_true')
parser.add_argument('--no-shuffle', action='store_false', dest='shuffle', default=True)

# Model
parser.add_argument('--backbone', choices=['small-resnet50v2', 'resnet50v2', 'resnet50', 'affine'])
parser.add_argument('--feat-norm', choices=['l2', 'bn'])
parser.add_argument('--proj-norm', choices=['l2', 'bn', 'sn'])
parser.add_argument('--proj-dim', type=int, default=128)
parser.add_argument('--stop-gradient', action='store_true')

# Loss objective
parser.add_argument('--loss', choices=['ce', 'supcon', 'hiercon', 'hiercon2', 'simclr', 'no-op'], default='ce')
parser.add_argument('--temp', type=float, default=0.1)
parser.add_argument('--weight-decay', type=float, default=1e-4)

# Training hyperparameters
parser.add_argument('--optimizer', choices=['sgd', 'adam', 'lamb'], default='sgd')

parser.add_argument('--init-epoch', type=int, default=0)
parser.add_argument('--epochs', type=int)
parser.add_argument('--steps-exec', type=int, help='steps per execution')
parser.add_argument('--train-steps', type=int, help='train steps per epoch')
parser.add_argument('--val-steps', type=int, help='val steps per epoch')

parser.add_argument('--bsz', type=int)
parser.add_argument('--warmup', type=int, default=0)
parser.add_argument('--lr', type=float)
parser.add_argument('--lr-decays', type=int, nargs='+', help='decays learning rate at the specified epochs')
parser.add_argument('--cosine-decay', action='store_true')

parser.add_argument('--recompile', action='store_true')

# Strategy
parser.add_argument('--tpu', action='store_true')
parser.add_argument('--multi-cpu', action='store_true')
parser.add_argument('--policy', choices=['mixed_bfloat16', 'float32'], default='float32')

# Other
parser.add_argument('--load', action='store_true')
parser.add_argument('--tsne', action='store_true')
parser.add_argument('--base-dir', type=str, default='out/')
parser.add_argument('--log-level', choices=['debug', 'info', 'warning', 'error'], default='info')
parser.add_argument('--no-save', action='store_true', help='skip saving logs and model checkpoints')
parser.add_argument('--profile-batch', type=int, nargs='*', default=0)

# Tensorboard
parser.add_argument('--update-freq', type=str, default='epoch', help='tensorboard metrics update frequency')


def setup(args):
    # Logging
    logging.getLogger().setLevel(args.log_level.upper())

    # Removed duplicate stream in TF logger
    tf_logger = logging.getLogger('tensorflow')
    for h in tf_logger.handlers:
        tf_logger.removeHandler(h)

    # Output directory
    args.out = os.path.join(args.base_dir, args.loss, args.data_id, f'{args.backbone}-{args.feat_norm}')
    logging.info(f"out directory: '{args.out}'")
    if not args.load:
        if args.out.startswith('gs://'):
            os.system(f"gsutil -m rm {os.path.join(args.out, '**')}")
        else:
            if os.path.exists(args.out):
                shutil.rmtree(args.out)
            os.makedirs(args.out)
        logging.info(f"cleared any previous work in '{args.out}'")

    # Strategy
    if args.tpu:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.TPUStrategy(resolver)
    elif len(tf.config.list_physical_devices('GPU')) > 1:
        strategy = tf.distribute.MirroredStrategy()
    elif args.multi_cpu:
        strategy = tf.distribute.MirroredStrategy(['CPU:0', 'CPU:1'])
    else:
        strategy = tf.distribute.get_strategy()

    # Mixed precision
    policy = mixed_precision.Policy(args.policy)
    mixed_precision.set_global_policy(policy)

    # Dataset arguments
    args.views, args.with_batch_sims = ['image', 'image2'], True

    return strategy


def prepare_tensorboard_dev_logs(args):
    # Copy log files to local disk
    log_dir = os.path.join(args.out, 'logs')
    if log_dir.startswith('gs://'):
        logging.info('downloading log data from GCS')
        import shutil
        shutil.rmtree('./out/logs', ignore_errors=True)
        os.system(f"gsutil -m cp -r {log_dir} ./out")
        tensorboard_logdir = './out/logs'
    else:
        tensorboard_logdir = log_dir

    tensorboard_cmd = "tensorboard dev upload " \
                      f"--logdir {tensorboard_logdir} " \
                      f"--name '{args.data_id}, {args.loss}' " \
                      f"--description '{args.bsz} bsz, {args.feat_norm} feat norm, {args.proj_norm} proj norm' " \
                      "--one_shot "

    # Print command to upload tensorboard data
    print('=' * 40, end='\n\n')
    print(tensorboard_cmd)
    print('\n' + '=' * 40)

    return tensorboard_cmd


all_custom_objects = {**custom_losses.custom_objects, **custom_layers.custom_objects, **lr_schedule.custom_objects}


def set_epoch_steps(args, ds_info):
    if args.train_steps is None:
        args.train_steps = ds_info.splits['train'].num_examples // args.bsz
        logging.info(f'train_steps not specified. setting it to train_size // bsz = {args.train_steps}')
    if args.val_steps is None:
        for split_name in ['validation', 'test', 'train']:
            if split_name in ds_info.splits:
                args.val_steps = ds_info.splits[split_name].num_examples // args.bsz
                logging.info(f'val_steps not specified. setting it to val_size // bsz = {args.val_steps}')
                break


def load_augment_configs(args):
    if args.autoaugment:
        autoaugment = augmentations.AutoAugment()
        augment_fn = lambda x: autoaugment.distort(tf.image.random_flip_left_right(x))
    else:
        augment_fn = tf.image.random_flip_left_right

    first_view_train_config = augmentations.ViewConfig(name='image', rand_crop=True, augment_fn=augment_fn)
    second_view_train_config = augmentations.ViewConfig(name='image2', rand_crop=True, augment_fn=augment_fn)

    first_view_val_config = augmentations.ViewConfig(name='image', rand_crop=False, augment_fn=None)
    second_view_val_config = augmentations.ViewConfig(name='image2', rand_crop=True, augment_fn=augment_fn)

    view_train_configs = [first_view_train_config, second_view_train_config]
    view_val_configs = [first_view_val_config, second_view_val_config]

    augment_train_config = augmentations.AugmentConfig(view_train_configs)
    augment_val_config = augmentations.AugmentConfig(view_val_configs)

    return augment_train_config, augment_val_config
