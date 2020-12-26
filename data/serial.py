import tensorflow as tf


def serialize_example(img, label):
    serialized_img = tf.io.encode_png(img)
    features = {
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[serialized_img.numpy()])),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
    }
    example = tf.train.Example(features=tf.train.Features(feature=features))
    return example.SerializeToString()


def tf_serialize_example(img, label):
    return tf.py_function(serialize_example, (img, label), tf.string)


def parse_example(serial):
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64)
    }
    example = tf.io.parse_example(serial, feature_description)
    image = tf.io.decode_png(example['image'])
    label = tf.cast(example['label'], tf.uint16)
    return image, label


def ds_to_tfrecord(ds, path):
    serialized_ds = ds.map(tf_serialize_example, num_parallel_calls=tf.data.AUTOTUNE)

    writer = tf.data.experimental.TFRecordWriter(path)
    writer.write(serialized_ds)


def ds_from_tfrecord(train_paths, test_paths):
    train_serialized = tf.data.TFRecordDataset(train_paths)
    test_serialized = tf.data.TFRecordDataset(test_paths)

    ds_train = train_serialized.map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = test_serialized.map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)
    return ds_train, ds_test
