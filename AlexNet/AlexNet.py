import tensorflow as tf
from tensorflow.python.keras import layers

tf.executing_eagerly()


def _argment_helper(image):
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [128, 128, 3])
    image = tf.image.resize(image, [227, 227])
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
    dataset = dataset.map(map_func=parse_fn, num_parallel_calls=2)
    dataset = dataset.prefetch(buffer_size=62)
    dataset = dataset.batch(batch_size=32)
    return dataset


def alex_net():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), padding='valid',
                            activation='relu', input_shape=(227, 227, 3), kernel_initializer='uniform'))
    model.add(layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding='same',
                            activation='relu', kernel_initializer='uniform'))
    model.add(layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='valid',
                            activation='relu', kernel_initializer='uniform'))
    model.add(layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='valid',
                            activation='relu', kernel_initializer='uniform'))
    model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='valid',
                            activation='relu', kernel_initializer='uniform'))
    model.add(layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation=tf.keras.activations.relu))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4096, activation=tf.keras.activations.relu))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation=tf.keras.activations.softmax))
    return model


model = alex_net()
model.compile(optimizer=tf.keras.optimizers.SGD(0.1),
              loss='categorical_crossentropy', metrics=['accuracy'])
parsed_dataset = input_fn()

model.fit(parsed_dataset, epochs=20, steps_per_epoch=100, validation_data=parsed_dataset)
print(model.evaluate(input_fn()))
