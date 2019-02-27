> 注意：论文中使用的输入图片的shape是`32*32*1`，而代码中使用的是`28*28*1`。
# LeNet
LeNet是一种典型的卷积神经网络的结构，由Yann LeCun发明。它的网路结构如下图：
![LeNet](http://daweiwong.com/2017/03/07/MNIST%20LeNet-5/LeNet-5-structure.png)

代码实现参考下图结构：
![LeNet model](https://ask.qcloudimg.com/http-save/yehe-1881084/f3xo7y48br.png?imageView2/2/w/1620)
LeNet-5网络是针对灰度图进行训练的，输入图像大小为`32*32*1`

# Result
``` python
Train on 55000 samples, validate on 10000 samples
Epoch 1/10
55000/55000 [==============================] - 6s 112us/step - loss: 0.4190 - categorical_accuracy: 0.8775 - val_loss: 0.3363 - val_categorical_accuracy: 0.9057
Epoch 2/10
55000/55000 [==============================] - 5s 86us/step - loss: 0.3558 - categorical_accuracy: 0.8982 - val_loss: 0.3273 - val_categorical_accuracy: 0.9066
Epoch 3/10
55000/55000 [==============================] - 5s 83us/step - loss: 0.3455 - categorical_accuracy: 0.9012 - val_loss: 0.3228 - val_categorical_accuracy: 0.9058
Epoch 4/10
55000/55000 [==============================] - 5s 82us/step - loss: 0.3394 - categorical_accuracy: 0.9030 - val_loss: 0.3241 - val_categorical_accuracy: 0.9103
Epoch 5/10
55000/55000 [==============================] - 5s 83us/step - loss: 0.3324 - categorical_accuracy: 0.9050 - val_loss: 0.3210 - val_categorical_accuracy: 0.9087
Epoch 6/10
55000/55000 [==============================] - 5s 83us/step - loss: 0.3303 - categorical_accuracy: 0.9046 - val_loss: 0.3125 - val_categorical_accuracy: 0.9119
Epoch 7/10
55000/55000 [==============================] - 5s 83us/step - loss: 0.3265 - categorical_accuracy: 0.9071 - val_loss: 0.3005 - val_categorical_accuracy: 0.9158
Epoch 8/10
55000/55000 [==============================] - 5s 83us/step - loss: 0.3229 - categorical_accuracy: 0.9087 - val_loss: 0.3090 - val_categorical_accuracy: 0.9114
Epoch 9/10
55000/55000 [==============================] - 5s 83us/step - loss: 0.3199 - categorical_accuracy: 0.9092 - val_loss: 0.3102 - val_categorical_accuracy: 0.9128
Epoch 10/10
55000/55000 [==============================] - 5s 83us/step - loss: 0.3189 - categorical_accuracy: 0.9101 - val_loss: 0.2883 - val_categorical_accuracy: 0.9173
10000/10000 [==============================] - 0s 36us/step
loss:0.288287, accuracy:0.917300
```

### 参考文章
- @BookThief:[卷积神经网络 LeNet-5各层参数详解](https://www.jianshu.com/p/ce609f9b5910)
