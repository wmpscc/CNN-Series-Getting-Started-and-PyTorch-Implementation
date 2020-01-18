import tensorflow as tf
from tensorflow.python.keras import layers
import cv2

tf.executing_eagerly()


def _argment_helper(image):
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [128, 128, 3])
    image = tf.math.divide(image, tf.constant(255.0))
    return image


def parse_fn(example_proto):
    "Parse TFExample records and perform simple data augmentation."
    image_feature_description = {
        'label': tf.io.FixedLenFeature([], tf.int64),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
    }
    parsed = tf.io.parse_single_example(example_proto, image_feature_description)
    image = tf.image.decode_jpeg(parsed['image_raw'], 3)
    image = _argment_helper(image)
    parsed['label'] = tf.cast(parsed['label'], tf.int64)
    y = tf.one_hot(parsed['label'], 10)
    return image, y


def input_fn():
    dataset = tf.data.TFRecordDataset('train.tfrecords')
    dataset = dataset.shuffle(buffer_size=512)
    dataset = dataset.repeat()
    dataset = dataset.map(map_func=parse_fn, num_parallel_calls=2)
    dataset = dataset.prefetch(buffer_size=62)
    dataset = dataset.batch(batch_size=32)
    return dataset


def net():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', strides=(2, 2),
                            input_shape=(128, 128, 3), kernel_initializer='uniform'))
    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', strides=(2, 2),
                            kernel_initializer='uniform'))
    model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same', strides=(2, 2),
                            kernel_initializer='uniform'))
    model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same', strides=(2, 2),
                            kernel_initializer='uniform'))
    model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same', strides=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(units=1024, activation='relu'))
    model.add(layers.Dense(units=10, activation='softmax'))

    model.compile(optimizer=tf.keras.optimizers.SGD(0.1),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    parsed_dataset = input_fn()

    model.fit(parsed_dataset, epochs=20, validation_data=parsed_dataset)
    print(model.evaluate(input_fn()))


net()
