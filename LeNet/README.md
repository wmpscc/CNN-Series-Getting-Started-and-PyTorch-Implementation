> 注意：论文中使用的输入图片的shape是`32*32*1`，而代码中使用的是`28*28*1`。
# LeNet
LeNet是一种典型的卷积神经网络的结构，由Yann LeCun发明。它的网路结构如下图：
![LeNet](http://daweiwong.com/2017/03/07/MNIST%20LeNet-5/LeNet-5-structure.png)

代码实现参考下图结构：
![LeNet model](https://ask.qcloudimg.com/http-save/yehe-1881084/f3xo7y48br.png?imageView2/2/w/1620)
LeNet-5网络是针对灰度图进行训练的，输入图像大小为`32*32*1`

# Result
``` python
training on  cuda
epoch 1, loss 1.8365, train acc 0.329, test acc 0.596, time  4.9 sec
epoch 2, loss 0.4777, train acc 0.634, test acc 0.680, time  4.3 sec
epoch 3, loss 0.2691, train acc 0.699, test acc 0.694, time  5.8 sec
epoch 4, loss 0.1847, train acc 0.721, test acc 0.727, time  4.4 sec
epoch 5, loss 0.1378, train acc 0.736, test acc 0.739, time  3.9 sec
epoch 6, loss 0.1077, train acc 0.748, test acc 0.749, time  3.9 sec
epoch 7, loss 0.0879, train acc 0.759, test acc 0.765, time  3.9 sec
epoch 8, loss 0.0734, train acc 0.770, test acc 0.774, time  4.0 sec
epoch 9, loss 0.0630, train acc 0.779, test acc 0.779, time  4.0 sec
epoch 10, loss 0.0548, train acc 0.787, test acc 0.785, time  4.0 sec
epoch 11, loss 0.0482, train acc 0.795, test acc 0.786, time  4.1 sec
epoch 12, loss 0.0430, train acc 0.802, test acc 0.797, time  4.3 sec
epoch 13, loss 0.0385, train acc 0.809, test acc 0.807, time  4.0 sec
epoch 14, loss 0.0347, train acc 0.816, test acc 0.800, time  4.0 sec
epoch 15, loss 0.0317, train acc 0.822, test acc 0.810, time  4.3 sec
epoch 16, loss 0.0289, train acc 0.827, test acc 0.814, time  4.1 sec
epoch 17, loss 0.0268, train acc 0.831, test acc 0.821, time  4.2 sec
epoch 18, loss 0.0247, train acc 0.836, test acc 0.824, time  4.1 sec
epoch 19, loss 0.0231, train acc 0.838, test acc 0.823, time  4.2 sec
epoch 20, loss 0.0216, train acc 0.839, test acc 0.826, time  4.2 sec
```

### 参考文章
- @BookThief:[卷积神经网络 LeNet-5各层参数详解](https://www.jianshu.com/p/ce609f9b5910)
