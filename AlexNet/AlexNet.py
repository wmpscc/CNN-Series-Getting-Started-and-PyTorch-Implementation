import tensorflow as tf
from tensorflow._api.v1.keras import layers


# Food 101 数据集

def _parse_function(serialized_example_test):
    features = tf.parse_single_example(
        serialized_example_test,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string),
        }
    )
    img_train = features['image_raw']
    # image_decoded = tf.decode_raw(img_train, tf.uint8)
    image_decoded = tf.image.decode_image(img_train, channels=3)
    image_resized = tf.image.resize_images(image_decoded, [227, 227])
    labels = tf.cast(features['label'], tf.int64)
    labels = tf.one_hot(labels, 101)
    shape = tf.cast([227, 227], tf.int32)
    return image_resized, labels


def alex_net():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), padding='valid',
                            activation='relu', input_shape=(227, 227, 3)))
    model.add(layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding='same',
                            activation='relu'))
    model.add(layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='valid',
                            activation='relu'))
    model.add(layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='valid',
                            activation='relu'))
    model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='valid',
                            activation='relu'))
    model.add(layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation=tf.keras.activations.relu))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4096, activation=tf.keras.activations.relu))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(101, activation=tf.keras.activations.softmax))
    return model


filenames = ["/home/heolis/Data/food-101/TFRecord/train0.tfrecords",
             "/home/heolis/Data/food-101/TFRecord/train1.tfrecords",
             "/home/heolis/Data/food-101/TFRecord/train2.tfrecords",
             "/home/heolis/Data/food-101/TFRecord/train3.tfrecords"]
trainSet = tf.data.TFRecordDataset(filenames)
trainSet = trainSet.map(_parse_function)
trainSet = trainSet.repeat(10)
trainSet = trainSet.batch(32)
iterator_train = trainSet.make_one_shot_iterator()

filenames = ["/home/heolis/Data/food-101/TFRecord/train3.tfrecords"]
testSet = tf.data.TFRecordDataset(filenames)
testSet = testSet.map(_parse_function)
testSet = testSet.repeat(10)
testSet = testSet.batch(32)
iterator_test = testSet.make_one_shot_iterator()

model = alex_net()

model.compile(optimizer=tf.train.GradientDescentOptimizer(0.03),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=[tf.keras.metrics.categorical_accuracy])

model.fit(iterator_train, epochs=10, validation_data=testSet, steps_per_epoch=10000)

loss, accuracy = model.evaluate(testSet)
print("loss:%f, accuracy:%f" % (loss, accuracy))
