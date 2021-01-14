import tensorflow as tf


def scale_min_dim(img, imsize):
    imshape = tf.cast(tf.shape(img), tf.float32)
    h, w = imshape[0], imshape[1]
    small_length = tf.minimum(h, w)
    scale = tf.cast(imsize + 1, tf.float32) / small_length
    new_size = [tf.cast(h * scale, tf.int32), tf.cast(w * scale, tf.int32)]
    img = tf.image.resize(img, new_size)
    return img


def min_scale_rand_crop(img, imsize):
    img = scale_min_dim(img, imsize)
    img = tf.image.random_crop(img, [imsize, imsize, 3])
    img = tf.cast(img, tf.uint8)
    return img


def min_scale_center_crop(img, imsize):
    img = scale_min_dim(img, imsize)
    img = tf.image.resize_with_crop_or_pad(img, imsize, imsize)
    img = tf.cast(img, tf.uint8)
    return img


def color_augment(image):
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
    return image