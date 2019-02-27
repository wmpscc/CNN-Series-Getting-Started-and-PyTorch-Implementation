import tensorflow as tf
from tensorflow._api.v1.keras import layers
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

model = tf.keras.Sequential()
model.add(layers.Dense(64, bias_regularizer=tf.keras.regularizers.l2(0.01), activation='relu'))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(64, bias_regularizer=tf.keras.regularizers.l2(0.01), activation='relu'))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer=tf.train.AdagradOptimizer(0.3),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=[tf.keras.metrics.categorical_accuracy])


model.fit(mnist.train.images, mnist.train.labels, epochs=10, batch_size=32,
          validation_data=(mnist.test.images, mnist.test.labels))


