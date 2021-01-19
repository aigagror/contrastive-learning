import argparse

import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.python.tpu import tpu_function

parser = argparse.ArgumentParser()

# Data
parser.add_argument('--data', choices=['cifar10', 'imagenet'])

# CNN
parser.add_argument('--cnn', choices=['small-resnet50v2', 'resnet50v2'])
parser.add_argument('--norm-feats', action='store_true')

# Method
parser.add_argument('--method', choices=['ce', 'supcon', 'supcon-pce'])

# Training hyperparameters
parser.add_argument('--init-epoch', type=int, default=0)
parser.add_argument('--epochs', type=int)
parser.add_argument('--bsz', type=int)
parser.add_argument('--lr', type=float)
parser.add_argument('--lr-decays', type=int, nargs='+', help='decays learning rate at the specified epochs')
parser.add_argument('--l2-reg', type=float, default=1e-4)
parser.add_argument('--spe', type=int, help='steps per execution')
parser.add_argument('--train-steps', type=int, help='train steps per epoch')
parser.add_argument('--val-steps', type=int, help='val steps per epoch')

# Strategy
parser.add_argument('--tpu', action='store_true')
parser.add_argument('--policy', choices=['mixed_bfloat16', 'float32'], default='float32')

# Other
parser.add_argument('--load', action='store_true')
parser.add_argument('--tsne', action='store_true')
parser.add_argument('--out', type=str, default='out/')
parser.add_argument('--debug', action='store_true')

def cross_replica_concat(tensor):
  """A cross-replica concatenation of a single Tensor across TPU cores.
  Input tensor is assumed to have batch dimension as the first dimension. The
  concatenation is done along the batch dimension.
  Args:
    tensor: Input Tensor which should be concatenated across TPU cores.
  Returns:
    The concatenated Tensor with batch dimension multiplied by the number of
      TPU cores.
  """
  num_tpu_replicas = tpu_function.get_tpu_context().number_of_shards

  if num_tpu_replicas is not None:
    # Scattered tensor has shape [num_replicas, local_batch_size, ...]
    scattered_tensor = tf.scatter_nd(
        indices=[[local_tpu_replica_id()]],
        updates=[tensor],
        shape=[num_tpu_replicas] + tensor.shape.as_list())
    reduced_tensor = tf.tpu.cross_replica_sum(scattered_tensor)
    # Returned tensor has shape [num_replicas * local_batch_size, ...]
    return tf.reshape(reduced_tensor,
                      [-1] + scattered_tensor.shape.as_list()[2:])
  else:
    # This is a no op if not running on TPU
    return tensor


def setup(args):
    # Logging
    tf.get_logger().setLevel('WARNING')

    # Strategy
    if args.tpu:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(resolver)
        # This is the TPU initialization code that has to be at the beginning.
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.TPUStrategy(resolver)
    elif len(tf.config.list_physical_devices('GPU')) > 1:
        strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = tf.distribute.get_strategy()

    # Mixed precision
    policy = mixed_precision.Policy(args.policy)
    mixed_precision.set_global_policy(policy)

    for dtype in ['bfloat16', 'float32']:
        if dtype in args.policy:
            args.dtype = dtype
            break

    return strategy
