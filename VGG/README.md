# VGG16
 VGGNet是由牛津大学计算机视觉组和Google DeepMind项目的研究员共同研发的卷积神经网络模型，包含VGG16和VGG19两种模型，其网络模型如图所示。
![vgg16](https://rescdn.mdpi.cn/remotesensing/remotesensing-10-00351/article_deploy/html/images/remotesensing-10-00351-g004.png)
从网络模型可以看出，VGG16相比AlexNet类的模型具有较深的深度，通过反复堆叠`3*3`的卷积层和`2*2`的池化层，VGG16构建了较深层次的网络结构，整个网络的卷积核使用了一致的`3*3`的尺寸，最大池化层尺寸也一致为`2*2`。与AlexNet主要有以下不同：

- Vgg16有16层网络，AlexNet只有8层；
- 在训练和测试时使用了多尺度做数据增强。

## 参考文献
- [VGG16网络模型](http://ethereon.github.io/netscope/#/gist/dc5003de6943ea5a6b8b)
- @冲弱:[卷积神经网络模型解读汇总——LeNet5，AlexNet、ZFNet、VGG16、GoogLeNet和ResNet](https://juejin.im/post/5ae283c4f265da0b886d2323)
