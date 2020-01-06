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


class DenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_conv):
        super(DenseBlock, self).__init__()
        self.out_channels = out_channels

        layers = []
        for i in range(num_conv):
            in_c = in_channels + i * self.out_channels
            layers.append(self.conv_block(in_c, self.out_channels))

        self.net = nn.ModuleList(layers)
        self.out_channels = in_channels + num_conv * out_channels

    def conv_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )
        return block

    def forward(self, X):
        for layer in self.net:
            Y = layer(X)
            X = torch.cat((X, Y), dim=1)
        return X


class DenseNet(nn.Module):
    def __init__(self, classes):
        super(DenseNet, self).__init__()
        self.classes = classes
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        num_channels, growth_rate = 64, 32  # 当前通道数64,每层增加32通道(即每层卷积输出通道为32)
        num_convs_in_dense_block = [4, 4, 4, 4]  # 每个dense block 卷积数

        for i, num_convs in enumerate(num_convs_in_dense_block):
            block = DenseBlock(num_channels, growth_rate, num_convs)
            self.net.add_module("dense_block_%d" % i, block)

            num_channels = block.out_channels  # 上一个块的输出通道数

            if i != len(num_convs_in_dense_block) - 1:
                self.net.add_module("trainsition_block_%d" % i, self.transition_block(num_channels, num_channels // 2))
                num_channels = num_channels // 2

        self.net.add_module("BN", nn.BatchNorm2d(num_channels))
        self.net.add_module("relu", nn.ReLU())
        self.net.add_module("global_avg_pool", GlobalAvgPool2d())
        self.net.add_module("flatten", FlattenLayer())
        self.net.add_module("fc", nn.Linear(num_channels, self.classes))

    def forward(self, X):
        return self.net(X)

    def transition_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        return block


if __name__ == "__main__":
    INPUT_SIZE = 224
    BATCH_SIZE = 16
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    denseNet = DenseNet(classes=10)
    denseNet = denseNet.to(device)
    optimizer = optim.Adam(denseNet.parameters(), lr=1e-3)
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
            y_pred = denseNet(X)
            loss = lossFN(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_loss += loss.cpu().item()
            sum_acc += (y_pred.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(valDL, denseNet)
        print("epoch %d: loss=%.4f \t acc=%.4f \t test acc=%.4f" % (epoch + 1, sum_loss / n, sum_acc / n, test_acc))
