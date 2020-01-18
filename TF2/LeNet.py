import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.utils.np_utils import to_categorical

# 准备数据集
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = tf.cast(X_train.reshape(-1, 28, 28, 1), tf.float32)
X_test = tf.cast(X_test.reshape(-1, 28, 28, 1), tf.float32)
y_train = tf.convert_to_tensor(y_train)
y_train = to_categorical(y_train)
y_test = tf.convert_to_tensor(y_test)
y_test = to_categorical(y_test)
print(y_train)

# LeNet模型
model = tf.keras.Sequential()
model.add(layers.Conv2D(filters=32, kernel_size=(5, 5), padding='valid', input_shape=(28, 28, 1), strides=(1, 1),
                        kernel_initializer='uniform', activation='relu'))
model.add(layers.MaxPool2D(strides=(2, 2), padding='same'))
model.add(layers.Conv2D(filters=64, kernel_size=(5, 5), padding='valid', strides=(1, 1), activation='relu',
                        kernel_initializer='uniform'))
model.add(layers.MaxPool2D(strides=(2, 2), padding='same'))
model.add(layers.Flatten())
model.add(layers.Dense(units=100))
model.add(layers.Dense(units=10, activation='softmax'))

model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()
model.fit(X_train, y_train, epochs=10, batch_size=32,
          validation_data=(X_test, y_test), verbose=1)

loss, accuracy = model.evaluate(X_test, y_test)
print("loss:%f, accuracy:%f" % (loss, accuracy))
