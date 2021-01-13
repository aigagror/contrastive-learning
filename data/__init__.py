import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.python.data import AUTOTUNE


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


def scale_min_dim(img, imsize):
    imshape = tf.cast(tf.shape(img), tf.float32)
    h, w = imshape[0], imshape[1]
    small_length = tf.minimum(h, w)
    scale = tf.cast(imsize + 1, tf.float32) / small_length
    new_size = [tf.cast(h * scale, tf.int32), tf.cast(w * scale, tf.int32)]
    img = tf.image.resize(img, new_size)
    return img


def rand_resize(img, imsize):
    img = scale_min_dim(img, imsize)
    img = tf.image.random_crop(img, [imsize, imsize, 3])
    img = tf.cast(img, tf.uint8)
    return img


def center_resize(img, imsize):
    img = scale_min_dim(img, imsize)
    img = tf.image.resize_with_crop_or_pad(img, imsize, imsize)
    img = tf.cast(img, tf.uint8)
    return img


def augment_img(image):
    # Random flip
    image = tf.image.random_flip_left_right(image)

    # Color Jitter
    if tf.random.uniform([]) < 0.8:
        image = tf.image.random_brightness(image, 0.4)
        image = tf.image.random_contrast(image, 0.6, 1.4)
        image = tf.image.random_saturation(image, 0.6, 1.4)
        image = tf.image.random_hue(image, 0.1)

    # Gray scale
    if tf.random.uniform([]) < 0.2:
        image = tf.image.rgb_to_grayscale(image)
        image = tf.repeat(image, 3, axis=-1)

    # Clip
    image = tf.clip_by_value(image, 0, 255)
    image = tf.cast(image, tf.uint8)
    return image


def load_datasets(args):
    if args.data == 'cifar10':
        imsize, nclass = 32, 10
        (x_train, y_train), (x_val, y_val) = datasets.cifar10.load_data()
        ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train.flatten()))
        ds_val = tf.data.Dataset.from_tensor_slices((x_val, y_val.flatten()))

        # Shuffle entire dataset
        ds_train = ds_train.shuffle(len(ds_train))
        ds_val = ds_val.shuffle(len(ds_val))

    elif args.data == 'imagenet':
        imsize, nclass = 224, 1000
        train_files = tf.data.Dataset.list_files('gs://aigagror/datasets/imagenet/train*', shuffle=True)
        val_files = tf.data.Dataset.list_files('gs://aigagror/datasets/imagenet/validation-*', shuffle=True)

        train_data = train_files.interleave(tf.data.TFRecordDataset, num_parallel_calls=AUTOTUNE)
        val_data = val_files.interleave(tf.data.TFRecordDataset, num_parallel_calls=AUTOTUNE)

        ds_train = train_data.map(parse_imagenet_example, AUTOTUNE)
        ds_val = val_data.map(parse_imagenet_example, AUTOTUNE)
    else:
        raise Exception(f'unknown data {args.data}')

    # Preprocess
    def process_train(img, label):
        ret = {'imgs': augment_img(rand_resize(img, imsize)), 'labels': label}
        if args.method.startswith('supcon'):
            ret['imgs2'] = augment_img(rand_resize(img, imsize))
        return ret

    def process_val(img, label):
        return {'imgs': center_resize(img, imsize), 'imgs2':  augment_img(center_resize(img, imsize)),
                'labels': label}

    ds_train = ds_train.map(process_train, AUTOTUNE)
    ds_val = ds_val.map(process_val, AUTOTUNE)

    # Batch and prefetch
    ds_train = ds_train.batch(args.bsz, drop_remainder=True).prefetch(AUTOTUNE)
    ds_val = ds_val.batch(args.bsz, drop_remainder=True).prefetch(AUTOTUNE)

    return ds_train, ds_val, nclass
