import tensorflow as tf
from tensorflow._api.v1.keras import layers
from tensorflow.examples.tutorials.mnist import input_data

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

filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
dataset = tf.data.TFRecordDataset(filenames)

model.fit(X_train, y_train, epochs=10, batch_size=32,
          validation_data=(X_test, y_test))

loss, accuracy = model.evaluate(X_test, y_test)
print("loss:%f, accuracy:%f" % (loss, accuracy))