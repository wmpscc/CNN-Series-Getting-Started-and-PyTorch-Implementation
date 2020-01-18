import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.utils.np_utils import to_categorical
tf.executing_eagerly()

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = tf.cast(X_train.reshape(-1, 28, 28, 1), tf.float32)
X_test = tf.cast(X_test.reshape(-1, 28, 28, 1), tf.float32)
y_train = tf.convert_to_tensor(y_train)
y_train = tf.one_hot(y_train, 10)
y_test = tf.convert_to_tensor(y_test)
y_test = tf.one_hot(y_test, 10)

model = tf.keras.Sequential()
model.add(layers.Dense(64, bias_regularizer=tf.keras.regularizers.l2(0.01), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(64, bias_regularizer=tf.keras.regularizers.l2(0.01), activation='relu'))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer=tf.keras.optimizers.SGD(0.1),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32)

loss, accuracy = model.evaluate(X_test, y_test)
print("loss:%f, accuracy:%f" % (loss, accuracy))
