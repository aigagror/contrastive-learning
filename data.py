import tensorflow as tf
from tensorflow.keras import datasets, preprocessing
from tensorflow.python.data import AUTOTUNE


def augment_img(image):
    # Crop
    imsize = image.shape[0]
    rand_scale = tf.random.uniform([], 1, 2)
    rand_size = tf.round(rand_scale * imsize)
    image = tf.image.resize(image, [rand_size, rand_size])
    image = tf.image.random_crop(image, [imsize, imsize, 3])

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
        image = tf.tile(image, [1, 1, 3])

    # Clip
    image = tf.clip_by_value(image, 0, 1)

    return image


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
    img = tf.io.decode_image(example['image/encoded'], channels=3)
    label = example['image/class/label'] - 1
    return img, label


def load_datasets(args, strategy):
    if args.data == 'cifar10':
        imsize, nclass = 32, 10
        (x_train, y_train), (x_val, y_val) = datasets.cifar10.load_data()
        ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train.flatten())).cache()
        ds_val = tf.data.Dataset.from_tensor_slices((x_val, y_val.flatten())).cache()

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

    # Map functions
    def cast_resize(img, labels):
        img = tf.image.convert_image_dtype(img, args.dtype)
        return preprocessing.image.smart_resize(img, [imsize, imsize]), labels

    # Preprocess
    ds_train = ds_train.map(cast_resize, num_parallel_calls=AUTOTUNE)
    ds_val = ds_val.map(cast_resize, num_parallel_calls=AUTOTUNE)

    if args.method.startswith('supcon'):
        def dual_augment(imgs, labels):
            return augment_img(imgs), augment_img(imgs), labels

        def dual_views(imgs, labels):
            return imgs, imgs, labels

        ds_train = ds_train.map(dual_augment, num_parallel_calls=AUTOTUNE)
        ds_val = ds_val.map(dual_views, num_parallel_calls=AUTOTUNE)
    else:
        def augment(image, labels):
            return augment_img(image), labels

        ds_train = ds_train.map(augment, num_parallel_calls=AUTOTUNE)

    # Batch and prefetch
    ds_train = ds_train.batch(args.bsz, drop_remainder=True).prefetch(AUTOTUNE)
    ds_val = ds_val.batch(args.bsz, drop_remainder=True).prefetch(AUTOTUNE)

    # Distribute among strategy
    ds_train = strategy.experimental_distribute_dataset(ds_train)
    ds_val = strategy.experimental_distribute_dataset(ds_val)

    return ds_train, ds_val, nclass
