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


class Inception(nn.Module):
    # c1-c4为四条线路输出的通道数
    def __init__(self, in_c, c1, c2, c3, c4):
        super(Inception, self).__init__()
        self.p1_1 = nn.Conv2d(in_c, c1, kernel_size=1)

        self.p2_1 = nn.Conv2d(in_c, c2[0], 1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)

        self.p3_1 = nn.Conv2d(in_c, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)

        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_c, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu((self.p2_1(x)))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        return torch.cat((p1, p2, p3, p4), dim=1)


class GoogleNet(nn.Module):
    def __init__(self, classes):
        self.classes = classes
        super(GoogleNet, self).__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.b3 = nn.Sequential(
            Inception(192, 64, (96, 128), (16, 32), 32),
            Inception(256, 128, (128, 192), (32, 96), 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.b4 = nn.Sequential(
            Inception(480, 192, (96, 208), (16, 48), 64),
            Inception(512, 160, (112, 224), (24, 64), 64),
            Inception(512, 128, (128, 256), (24, 64), 64),
            Inception(512, 112, (144, 288), (32, 64), 64),
            Inception(528, 256, (160, 320), (32, 128), 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.b5 = nn.Sequential(
            Inception(832, 256, (160, 320), (32, 128), 128),
            Inception(832, 384, (192, 384), (48, 128), 128),
            GlobalAvgPool2d(),
            nn.Dropout(0.4),
            FlattenLayer(),
            nn.Linear(1024, self.classes)
        )
        self.net = nn.Sequential(
            self.b1, self.b2, self.b3, self.b4, self.b5
        )

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    INPUT_SIZE = 224
    BATCH_SIZE = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    inceptionNet = GoogleNet(classes=10)
    inceptionNet = inceptionNet.to(device)
    optimizer = optim.Adam(inceptionNet.parameters(), lr=1e-3)
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
            y_pred = inceptionNet(X)
            loss = lossFN(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_loss += loss.cpu().item()
            sum_acc += (y_pred.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(valDL, inceptionNet)
        print("epoch %d: loss=%.4f \t acc=%.4f \t test acc=%.4f" % (epoch + 1, sum_loss / n, sum_acc / n, test_acc))
