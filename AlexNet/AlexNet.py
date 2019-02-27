import tensorflow as tf
from tensorflow._api.v1.keras import layers


# Food 101 数据集

def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string)
    image_resized = tf.image.resize_images(image_decoded, [227, 227, 3])
    return image_resized, label


def alex_net():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), padding='valid',
                            activation='relu'))
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
    model.add(layers.Dense(1000, activation=tf.keras.activations.softmax))
    return model


filenames = ["/media/data/oldcopy/PythonProject/Food101/TFRecord/train.tfrecords"]
trainSet = tf.data.TFRecordDataset(filenames)
trainSet = trainSet.map(_parse_function)
trainSet = trainSet.repeat()
trainSet = trainSet.batch(32)

filenames = ["/media/data/oldcopy/PythonProject/Food101/TFRecord/test.tfrecords"]
testSet = tf.data.TFRecordDataset(filenames)
testSet = testSet.map(_parse_function)
testSet = testSet.repeat()
testSet = testSet.batch(32)

model = alex_net()
model.fit(trainSet, epochs=10, batch_size=32, validation_data=testSet)

loss, accuracy = model.evaluate(testSet)
print("loss:%f, accuracy:%f" % (loss, accuracy))
