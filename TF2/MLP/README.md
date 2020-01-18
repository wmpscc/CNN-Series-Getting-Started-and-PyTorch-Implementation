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
55000/55000 [==============================] - 4s 77us/step - loss: 0.3974 - categorical_accuracy: 0.8776 - val_loss: 0.2439 - val_categorical_accuracy: 0.9309
Epoch 2/10
55000/55000 [==============================] - 3s 62us/step - loss: 0.1687 - categorical_accuracy: 0.9495 - val_loss: 0.1498 - val_categorical_accuracy: 0.9557
Epoch 3/10
55000/55000 [==============================] - 3s 62us/step - loss: 0.1315 - categorical_accuracy: 0.9602 - val_loss: 0.1331 - val_categorical_accuracy: 0.9607
Epoch 4/10
55000/55000 [==============================] - 3s 62us/step - loss: 0.1092 - categorical_accuracy: 0.9675 - val_loss: 0.1478 - val_categorical_accuracy: 0.9576
Epoch 5/10
55000/55000 [==============================] - 3s 62us/step - loss: 0.0938 - categorical_accuracy: 0.9714 - val_loss: 0.1253 - val_categorical_accuracy: 0.9636
Epoch 6/10
55000/55000 [==============================] - 4s 64us/step - loss: 0.0823 - categorical_accuracy: 0.9748 - val_loss: 0.1307 - val_categorical_accuracy: 0.9623
Epoch 7/10
55000/55000 [==============================] - 4s 71us/step - loss: 0.0721 - categorical_accuracy: 0.9773 - val_loss: 0.1364 - val_categorical_accuracy: 0.9638
Epoch 8/10
55000/55000 [==============================] - 3s 62us/step - loss: 0.0627 - categorical_accuracy: 0.9804 - val_loss: 0.1169 - val_categorical_accuracy: 0.9664
Epoch 9/10
55000/55000 [==============================] - 4s 64us/step - loss: 0.0547 - categorical_accuracy: 0.9830 - val_loss: 0.1265 - val_categorical_accuracy: 0.9668
Epoch 10/10
55000/55000 [==============================] - 4s 69us/step - loss: 0.0504 - categorical_accuracy: 0.9840 - val_loss: 0.1222 - val_categorical_accuracy: 0.9697
10000/10000 [==============================] - 0s 30us/step
loss:0.122211, accuracy:0.969700
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