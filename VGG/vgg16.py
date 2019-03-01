import tensorflow as tf
from tensorflow._api.v1.keras import layers


# Food 101 数据集

def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string)
    image_resized = tf.image.resize_images(image_decoded, [224, 224, 3])
    return image_resized, label


def vgg16(input_tensor, shape=(None, 224, 224, 3)):
    inputs = layers.Input(tensor=input_tensor, shape=shape)

    x = layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', name='conv1_1')(inputs)
    x = layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', name='conv1_2')(x)
    x = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name='pool1')(x)
    x = layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu', name='conv2_1')(x)
    x = layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu', name='conv2_2')(x)
    x = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name='pool2')(x)
    x = layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu', name='conv3_1')(x)
    x = layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu', name='conv3_2')(x)
    x = layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu', name='conv3_3')(x)
    x = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name='pool3')(x)
    x = layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu', name='conv4_1')(x)
    x = layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu', name='conv4_2')(x)
    x = layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu', name='conv4_3')(x)
    x = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name='pool4')(x)
    x = layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu', name='conv5_1')(x)
    x = layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu', name='conv5_2')(x)
    x = layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu', name='conv5_3')(x)
    x = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name='pool5')(x)
    x = layers.Flatten(name='flatten')(x)

    x = layers.Dense(units=4096, activation='relu', name='fc6')(x)
    x = layers.Dropout(0.5, name='drop6')(x)

    x = layers.Dense(units=4096, activation='relu', name='fc7')(x)
    x = layers.Dropout(0.5, name='drop7')(x)

    output = layers.Dense(units=1000, activation='softmax', name='fc8_prob')(x)
    model = tf.keras.Model(inputs=inputs, outputs=output, name='vgg16')
    return model


def train():
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

    model = vgg16(trainSet)
    model.fit(trainSet, epochs=10, batch_size=32, validation_data=testSet)

    loss, accuracy = model.evaluate(testSet)
    print("loss:%f, accuracy:%f" % (loss, accuracy))
