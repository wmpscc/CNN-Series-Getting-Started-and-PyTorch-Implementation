import torch
from torch import nn, optim
from torch.nn import functional as F
from utils.load_data_Fnt10 import load_data_Fnt10
from utils import evaluate_accuracy
from tqdm import tqdm


class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, X):
        return X.view(X.shape[0], -1)


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])


class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, same_shape=True):
        super(Residual, self).__init__()
        self.stride = 1 if same_shape else 2
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=self.stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if not same_shape:  # 通过1x1卷积核，步长为2.统一两个特征图shape，保证加法运算正常
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2)
        else:
            self.conv3 = None

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = F.relu(self.bn2(self.conv2(Y)))
        if self.conv3 is not None:
            X = self.conv3(X)
        return F.relu(X + Y)


class ResNet(nn.Module):
    def __init__(self, classes=1000):
        super(ResNet, self).__init__()
        self.classes = classes
        self.net = nn.Sequential()
        self.b1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.net.add_module("block1", self.b1)
        self.net.add_module("resnet_block_1", self.resnet_block(64, 64, 2, is_first_block=True))
        self.net.add_module("resnet_block_2", self.resnet_block(64, 128, 2))
        self.net.add_module("resnet_block_3", self.resnet_block(128, 256, 2))
        self.net.add_module("resnet_block_4", self.resnet_block(256, 512, 2))
        self.net.add_module("global_avg_pool", GlobalAvgPool2d())
        self.net.add_module("flatten", FlattenLayer())
        self.net.add_module('fc', nn.Linear(512, self.classes))

    def resnet_block(self, in_channels, out_channels, num_residuals, is_first_block=False):
        if is_first_block:
            assert in_channels == out_channels  # 整个模型的第一块的输入通道数等于输出通道数

        block = []
        for i in range(num_residuals):
            if i == 0 and not is_first_block:
                block.append(Residual(in_channels, out_channels, same_shape=False))
            else:
                block.append(Residual(out_channels, out_channels))  # 第一块输入通道数与输出通道数相等
        return nn.Sequential(*block)

    def forward(self, X):
        return self.net(X)


if __name__ == '__main__':
    INPUT_SIZE = 224
    BATCH_SIZE = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    resnet = ResNet(classes=10)
    resnet = resnet.to(device)
    optimizer = optim.Adam(resnet.parameters(), lr=1e-3)
    lossFN = nn.CrossEntropyLoss()

    trainDL, valDL = load_data_Fnt10(INPUT_SIZE, BATCH_SIZE)

    num_epochs = 10
    for epoch in range(num_epochs):
        sum_loss = 0
        sum_acc = 0
        batch_count = 0
        n = 0
        for X, y in tqdm(trainDL):
            X = X.to(device)
            y = y.to(device)
            y_pred = resnet(X)

            print(y_pred.size(), y.size())
            loss = lossFN(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_loss += loss.cpu().item()
            sum_acc += (y_pred.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(valDL, resnet)
        print("epoch %d: loss=%.4f \t acc=%.4f \t test acc=%.4f" % (epoch + 1, sum_loss / n, sum_acc / n, test_acc))
