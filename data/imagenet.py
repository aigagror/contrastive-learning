import tensorflow as tf
from tensorflow.python.data import AUTOTUNE

from data.data_utils import scale_min_dim, color_augment, min_scale_rand_crop


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
    img = tf.io.decode_image(example['image/encoded'], channels=3, expand_animations=False)
    label = example['image/class/label'] - 1
    return img, label


def augment_imagenet_img(image):
    """
    From original resnet paper
    https://arxiv.org/pdf/1512.03385.pdf
    :param image:
    :return:
    """

    # Random scale
    rand_size = tf.random.uniform([], 256, 481, tf.int32)
    image = scale_min_dim(image, rand_size)

    # Random crop
    image = tf.image.random_crop(image, [224, 224, 3])

    # Random flip
    image = tf.image.random_flip_left_right(image)

    # Color augment
    image = color_augment(image)

    # Clip
    image = tf.clip_by_value(image, 0, 255)
    image = tf.cast(image, tf.uint8)
    return image


def load_imagenet(args):
    imsize, nclass = 224, 1000
    train_files = tf.data.Dataset.list_files('gs://aigagror/datasets/imagenet/train*', shuffle=True)
    val_files = tf.data.Dataset.list_files('gs://aigagror/datasets/imagenet/validation-*', shuffle=True)
    train_data = train_files.interleave(tf.data.TFRecordDataset, num_parallel_calls=AUTOTUNE)
    val_data = val_files.interleave(tf.data.TFRecordDataset, num_parallel_calls=AUTOTUNE)
    ds_train = train_data.map(parse_imagenet_example, AUTOTUNE)
    ds_val = val_data.map(parse_imagenet_example, AUTOTUNE)

    # Preprocess
    def process_train(img, label):
        ret = {'imgs': augment_imagenet_img(img), 'labels': label}
        if args.method.startswith('supcon'):
            ret['imgs2'] = augment_imagenet_img(img)
        return ret

    def process_val(img, label):
        return {'imgs': min_scale_rand_crop(img, 224), 'imgs2': augment_imagenet_img(img), 'labels': label}

    ds_train = ds_train.map(process_train, AUTOTUNE)
    ds_val = ds_val.map(process_val, AUTOTUNE)
    info = {'nclass': nclass, 'input_shape': [imsize, imsize, 3]}
    return ds_train, ds_val, info
