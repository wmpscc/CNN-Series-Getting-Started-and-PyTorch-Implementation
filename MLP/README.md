# Multi-Layer Perceptron
> 用tf.keras实现多层感知机，目的是熟悉tf.keras训练模型的流程

### Environment
- Python 3.6.4
- TensorFlow 1.12.0
- tf.keras 2.1.6-tf
- MNIST Dataset

### Result
```python
Train on 55000 samples, validate on 10000 samples
Epoch 1/10
55000/55000 [==============================] - 4s 80us/step - loss: 0.3947 - categorical_accuracy: 0.8772 - val_loss: 0.2011 - val_categorical_accuracy: 0.9406
Epoch 2/10
55000/55000 [==============================] - 4s 74us/step - loss: 0.1759 - categorical_accuracy: 0.9474 - val_loss: 0.1594 - val_categorical_accuracy: 0.9515
Epoch 3/10
55000/55000 [==============================] - 4s 68us/step - loss: 0.1340 - categorical_accuracy: 0.9598 - val_loss: 0.1454 - val_categorical_accuracy: 0.9573
Epoch 4/10
55000/55000 [==============================] - 4s 65us/step - loss: 0.1109 - categorical_accuracy: 0.9655 - val_loss: 0.1270 - val_categorical_accuracy: 0.9635
Epoch 5/10
55000/55000 [==============================] - 3s 63us/step - loss: 0.0942 - categorical_accuracy: 0.9713 - val_loss: 0.1233 - val_categorical_accuracy: 0.9651
Epoch 6/10
55000/55000 [==============================] - 4s 64us/step - loss: 0.0825 - categorical_accuracy: 0.9744 - val_loss: 0.1190 - val_categorical_accuracy: 0.9667
Epoch 7/10
55000/55000 [==============================] - 3s 63us/step - loss: 0.0717 - categorical_accuracy: 0.9774 - val_loss: 0.1205 - val_categorical_accuracy: 0.9668
Epoch 8/10
55000/55000 [==============================] - 3s 61us/step - loss: 0.0630 - categorical_accuracy: 0.9801 - val_loss: 0.1256 - val_categorical_accuracy: 0.9662
Epoch 9/10
55000/55000 [==============================] - 3s 62us/step - loss: 0.0573 - categorical_accuracy: 0.9813 - val_loss: 0.1226 - val_categorical_accuracy: 0.9676
Epoch 10/10
55000/55000 [==============================] - 3s 61us/step - loss: 0.0519 - categorical_accuracy: 0.9837 - val_loss: 0.1188 - val_categorical_accuracy: 0.9691
```

# tf.keras
> [查看官方原文](https://tensorflow.google.cn/guide/keras)
## 构建简单模型
### 序列模型
```python
model = tf.keras.Sequential()
# Adds a densely-connected layer with 64 units to the model:
model.add(layers.Dense(64, activation='relu'))
# Add another:
model.add(layers.Dense(64, activation='relu'))
# Add a softmax layer with 10 output units:
model.add(layers.Dense(10, activation='softmax'))
```
### 配置层
```python
# Create a sigmoid layer:
layers.Dense(64, activation='sigmoid')
# Or:
layers.Dense(64, activation=tf.sigmoid)

# A linear layer with L1 regularization of factor 0.01 applied to the kernel matrix:
layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l1(0.01))

# A linear layer with L2 regularization of factor 0.01 applied to the bias vector:
layers.Dense(64, bias_regularizer=tf.keras.regularizers.l2(0.01))

# A linear layer with a kernel initialized to a random orthogonal matrix:
layers.Dense(64, kernel_initializer='orthogonal')

# A linear layer with a bias vector initialized to 2.0s:
layers.Dense(64, bias_initializer=tf.keras.initializers.constant(2.0))

```
## 训练和评估
### 设置训练流程
```python
model = tf.keras.Sequential([
# Adds a densely-connected layer with 64 units to the model:
layers.Dense(64, activation='relu'),
# Add another:
layers.Dense(64, activation='relu'),
# Add a softmax layer with 10 output units:
layers.Dense(10, activation='softmax')])

model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```
以下代码展示了配置模型以进行训练的几个示例：
```python
# Configure a model for mean-squared error regression.
model.compile(optimizer=tf.train.AdamOptimizer(0.01),
              loss='mse',       # mean squared error
              metrics=['mae'])  # mean absolute error

# Configure a model for categorical classification.
model.compile(optimizer=tf.train.RMSPropOptimizer(0.01),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=[tf.keras.metrics.categorical_accuracy])
```
### 输入 NumPy 数据
```python
import numpy as np

data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))

val_data = np.random.random((100, 32))
val_labels = np.random.random((100, 10))

model.fit(data, labels, epochs=10, batch_size=32,
          validation_data=(val_data, val_labels))
```

### 评估和预测
```python
data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))

model.evaluate(data, labels, batch_size=32)

model.evaluate(dataset, steps=30)
```

> 更多信息请参考官方文档
> https://tensorflow.google.cn/guide/keras