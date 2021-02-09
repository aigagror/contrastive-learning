import tensorflow as tf

CROP_PADDING = 32


def distorted_bounding_box_crop(image_bytes,
                                bbox,
                                min_object_covered=0.1,
                                aspect_ratio_range=(0.75, 1.33),
                                area_range=(0.05, 1.0),
                                max_attempts=100):
    """Generates cropped_image using one of the bboxes randomly distorted.

    See `tf.image.sample_distorted_bounding_box` for more documentation.

    Args:
      image_bytes: `Tensor` of binary image data.
      bbox: `Tensor` of bounding boxes arranged `[1, num_boxes, coords]`
          where each coordinate is [0, 1) and the coordinates are arranged
          as `[ymin, xmin, ymax, xmax]`. If num_boxes is 0 then use the whole
          image.
      min_object_covered: An optional `float`. Defaults to `0.1`. The cropped
          area of the image must contain at least this fraction of any bounding
          box supplied.
      aspect_ratio_range: An optional list of `float`s. The cropped area of the
          image must have an aspect ratio = width / height within this range.
      area_range: An optional list of `float`s. The cropped area of the image
          must contain a fraction of the supplied image within in this range.
      max_attempts: An optional `int`. Number of attempts at generating a cropped
          region of the image of the specified constraints. After `max_attempts`
          failures, return the entire image.
    Returns:
      cropped image `Tensor`
    """
    with tf.name_scope('distorted_bounding_box_crop'):
        shape = tf.image.extract_jpeg_shape(image_bytes)
        sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
            shape,
            bounding_boxes=bbox,
            min_object_covered=min_object_covered,
            aspect_ratio_range=aspect_ratio_range,
            area_range=area_range,
            max_attempts=max_attempts,
            use_image_if_no_bounding_boxes=True)
        bbox_begin, bbox_size, _ = sample_distorted_bounding_box

        # Crop the image to the specified bounding box.
        offset_y, offset_x, _ = tf.unstack(bbox_begin)
        target_height, target_width, _ = tf.unstack(bbox_size)
        crop_window = tf.stack([offset_y, offset_x, target_height, target_width])
        image = tf.image.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)

        return image


def _at_least_x_are_equal(a, b, x):
    """At least `x` of `a` and `b` `Tensors` are equal."""
    match = tf.equal(a, b)
    match = tf.cast(match, tf.int32)
    return tf.greater_equal(tf.reduce_sum(match), x)


def _decode_and_random_crop_jpg(image_bytes, image_size):
    """Make a random crop of image_size."""
    bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
    image = distorted_bounding_box_crop(image_bytes,
                                        bbox,
                                        min_object_covered=0.1,
                                        aspect_ratio_range=(3. / 4, 4. / 3.),
                                        area_range=(0.08, 1.0),
                                        max_attempts=10)
    original_shape = tf.image.extract_jpeg_shape(image_bytes)
    bad = _at_least_x_are_equal(original_shape, tf.shape(image), 3)

    image = tf.cond(
        bad,
        lambda: _decode_and_center_crop_jpg(image_bytes, image_size),
        lambda: tf.image.resize(image, [image_size, image_size], method='bicubic'))

    return image


def _decode_and_center_crop_jpg(image_bytes, image_size):
    """Crops to center of image with padding then scales image_size."""
    shape = tf.image.extract_jpeg_shape(image_bytes)
    image_height = shape[0]
    image_width = shape[1]

    padded_center_crop_size = tf.cast(
        ((image_size / (image_size + CROP_PADDING)) *
         tf.cast(tf.minimum(image_height, image_width), tf.float32)),
        tf.int32)

    offset_height = ((image_height - padded_center_crop_size) + 1) // 2
    offset_width = ((image_width - padded_center_crop_size) + 1) // 2
    crop_window = tf.stack([offset_height, offset_width,
                            padded_center_crop_size, padded_center_crop_size])
    image = tf.image.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)
    image = tf.image.resize([image], [image_size, image_size], method='bicubic')[0]

    return image


def _decode_and_crop_jpg(image_bytes, rand_crop, imsize):
    if rand_crop:
        image = _decode_and_random_crop_jpg(image_bytes, imsize)
    else:
        image = _decode_and_center_crop_jpg(image_bytes, imsize)
    return image


def _decode_png_and_crop(image_bytes, imsize, rand_crop):
    image = tf.image.decode_png(image_bytes)
    image = tf.image.pad_to_bounding_box(image, 4, 4, imsize + 8, imsize + 8)
    channels = tf.shape(image)[-1]
    if rand_crop:
        image = tf.image.random_crop(image, [imsize, imsize, channels])
    else:
        image = tf.image.resize(image, [imsize, imsize])
    return image


def process_encoded_example(image_bytes, label, imsize, channels, augment_config):
    inputs, targets = {}, {'label': label}
    for view_config in augment_config.view_configs:
        image = tf.cond(tf.image.is_jpeg(image_bytes),
                        lambda: _decode_and_crop_jpg(image_bytes, view_config.rand_crop, imsize),
                        lambda: _decode_png_and_crop(image_bytes, imsize, view_config.rand_crop))
        image = tf.cast(image, tf.uint8)

        # Augment
        image = view_config.augment(image)

        image = tf.ensure_shape(image, [imsize, imsize, channels])

        inputs[view_config.name] = image

    return inputs, targets
