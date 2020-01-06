import tensorflow as tf
tf.executing_eagerly()
import cv2


def _argment_helper(image):
    image = tf.cast(image, tf.float32)
    image = tf.math.divide(image, tf.constant(255.0))
    print(image.shape)
    return image


def parse_fn(example_proto):
    "Parse TFExample records and perform simple data augmentation."
    image_feature_description = {
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'depth': tf.io.FixedLenFeature([], tf.int64),
        'label': tf.io.FixedLenFeature([], tf.int64),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
    }
    parsed = tf.io.parse_single_example(example_proto, image_feature_description)
    image = tf.image.decode_image(parsed['image_raw'])
    parsed['image_raw'] = _argment_helper(image)
    print(parsed['depth'])
    return parsed


def input_fn():
    dataset = tf.data.TFRecordDataset('train.tfrecords')
    dataset = dataset.shuffle(buffer_size=224)
    dataset = dataset.map(map_func=parse_fn)
    dataset = dataset.batch(batch_size=32)
    dataset = dataset.prefetch(buffer_size=62)
    return dataset


p_dataset = input_fn()

for image_features in p_dataset:
    image_raw = image_features['image_raw'].numpy()
    # img = tf.image.decode_image(image_raw).numpy()
    # cv2.imshow("decode", image_raw)
    # cv2.waitKey(0)
    print(image_raw.shape)
