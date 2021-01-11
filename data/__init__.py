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


def resize(img, imsize, crop):
    # This smart resize function also casts images to float32 within the same 0-255 range.

    small_length = tf.reduce_min(tf.shape(img)[:2])
    scale = imsize / small_length
    img = tf.image.resize(img, [img.shape[0] * scale, img.shape[1] * scale])

    if crop == 'rand':
        img = tf.image.random_crop(img, [imsize, imsize, 3])
    else:
        assert crop == 'center'
        img = tf.image.resize_with_crop_or_pad(img, imsize, imsize)

    tf.debugging.assert_shapes([(img, [imsize, imsize, 3])])
    return img

def augment_img(image):
    # Random scale
    imshape = tf.shape(image)
    rand_scale = tf.random.uniform([], 1, 1.5)
    new_h = tf.round(rand_scale * tf.cast(imshape[0], tf.float32))
    new_w = tf.round(rand_scale * tf.cast(imshape[1], tf.float32))
    image = tf.image.resize(image, [new_h, new_w])
    image = tf.image.random_crop(image, [imshape[0], imshape[1], 3])

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
    image = tf.clip_by_value(image, 0, 255)
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
        img = tf.cast(img, args.dtype)
        ret = {'imgs': augment_img(resize(img, imsize, crop='rand')), 'labels': label}
        if args.method.startswith('supcon'):
            ret['imgs2'] = augment_img(resize(img, imsize, crop='rand'))
        return ret

    def process_val(img, label):
        img = tf.cast(img, args.dtype)
        ret = {'imgs': resize(img, imsize, crop='center'), 'labels': label}
        if args.method.startswith('supcon'):
            ret['imgs2'] = augment_img(resize(img, imsize, crop='center'))
        return ret

    ds_train = ds_train.map(process_train, AUTOTUNE)
    ds_val = ds_val.map(process_val, AUTOTUNE)

    # Batch and prefetch
    ds_train = ds_train.batch(args.bsz, drop_remainder=True).prefetch(AUTOTUNE)
    ds_val = ds_val.batch(args.bsz, drop_remainder=True).prefetch(AUTOTUNE)

    return ds_train, ds_val, nclass
