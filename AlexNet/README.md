# AlexNet
![AlexNet](https://images2018.cnblogs.com/blog/1160281/201711/1160281-20171124090206484-877449439.png)
- **1.贡献**：ILSVRC2012冠军，展现出了深度CNN在图像任务上的惊人表现，掀起CNN研究的热潮，是如今深度学习和AI迅猛发展的重要原因。ImageNet比赛为一直研究神经网络的Hinton提供了施展平台，AlexNet就是由hinton和他的两位学生发表的，在AlexNet之前，深度学习已经沉寂了很久。
- **2.网络结构**：如下图所示，8层网络，参数大约有60 million，使用了relu函数，头两个全连接层使用了0.5的dropout。使用了LRN和重叠的池化，现在LRN都不用了，一般用BN作Normalization。当时使用了多GPU训练。
- **3.预处理**：先down-sample成最短边为256的图像，然后剪出中间的256x256图像，再减均值做归一化（over training set）。 **训练时，做数据增强**，对每张图像，随机提取出227x227以及水平镜像版本的图像。除了数据增强，还使用了PCA对RGB像素降维的方式来缓和过拟合问题。
- **4.预测**：对每张图像提取出5张（四个角落以及中间）以及水平镜像版本，总共10张，平均10个预测作为最终预测。
- **5.超参数**：SGD，学习率0.01，batch size是128，momentum为0.9，weight decay为0.0005（论文有个权重更新公式），每当validation error不再下降时，学习率除以10。权重初始化用（0，0.01）的高斯分布，二四五卷积层和全连接层的bias初始化为1（给relu提供正值利于加速前期训练），其余bias初始化为0。

- #### AlexNet Architecture
![arch](img/architecture.png)

- #### AlexNet 每层的超参数及参数数量
![alex parma](https://upload-images.jianshu.io/upload_images/1689929-78bfb344b7f63729.png?imageMogr2/auto-orient/strip%7CimageView2/2)
##参考
- @PilgrimHui:[论文笔记：CNN经典结构1（AlexNet，ZFNet，OverFeat，VGG，GoogleNet，ResNet）](https://www.cnblogs.com/liaohuiqiang/p/9606901.html)
- @simshang:[AlexNet](http://simtalk.cn/2016/09/20/AlexNet/)

