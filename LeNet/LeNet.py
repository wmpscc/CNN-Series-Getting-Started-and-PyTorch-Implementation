import tensorflow as tf
from tensorflow._api.v1.keras import layers
from tensorflow.examples.tutorials.mnist import input_data

# 准备数据集
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
X_train = mnist.train.images
y_train = mnist.train.labels
X_test = mnist.test.images
y_test = mnist.test.labels

X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# LeNet模型
model = tf.keras.Sequential()
model.add(layers.Conv2D(filters=6, kernel_size=(5, 5), padding='valid', strides=(1, 1)))
model.add(layers.AvgPool2D(strides=(2, 2), padding='same'))
model.add(layers.Conv2D(filters=16, kernel_size=(5, 5), padding='valid', strides=(1, 1)))
model.add(layers.AvgPool2D(strides=(2, 2), padding='same'))
model.add(layers.Flatten())
model.add(layers.Dense(units=120))
model.add(layers.Dense(units=84))
model.add(layers.Dense(units=10, activation='softmax'))

model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=[tf.keras.metrics.categorical_accuracy])

model.fit(X_train, y_train, epochs=10, batch_size=32,
          validation_data=(X_test, y_test))

loss, accuracy = model.evaluate(X_test, y_test)
print("loss:%f, accuracy:%f" % (loss, accuracy))
