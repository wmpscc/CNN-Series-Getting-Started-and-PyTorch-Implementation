import torch
from torch import nn, optim
import torch.nn.functional as F
from utils.load_data_Fnt10 import load_data_Fnt10
from utils import evaluate_accuracy
from tqdm import tqdm


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])


class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, X):
        return X.view(X.shape[0], -1)


class NIN(nn.Module):
    def __init__(self, classes):
        self.classes = classes
        super(NIN, self).__init__()
        self.net = nn.Sequential(
            self.nin_block(3, 96, kernel_size=11, stride=4, padding=0),
            nn.MaxPool2d(kernel_size=3, stride=2),
            self.nin_block(96, 256, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            self.nin_block(256, 384, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout(0.5),
            self.nin_block(384, self.classes, kernel_size=3, stride=1, padding=1),
            GlobalAvgPool2d(),
            FlattenLayer()
        )

    def nin_block(self, in_channels, out_channels, kernel_size, stride, padding):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.ReLU()
        )
        return block

    def forward(self, img):
        return self.net(img)


if __name__ == '__main__':
    INPUT_SIZE = 224
    BATCH_SIZE = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ninNet = NIN(classes=10)
    ninNet = ninNet.to(device)
    optimizer = optim.Adam(ninNet.parameters(), lr=1e-3)
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
            y_pred = ninNet(X)
            loss = lossFN(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_loss += loss.cpu().item()
            sum_acc += (y_pred.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(valDL, ninNet)
        print("epoch %d: loss=%.4f \t acc=%.4f \t test acc=%.4f" % (epoch + 1, sum_loss / n, sum_acc / n, test_acc))
