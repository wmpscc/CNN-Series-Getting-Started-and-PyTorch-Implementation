import torch
from torch import nn, optim
from utils.load_data_Fnt10 import load_data_Fnt10
from utils import evaluate_accuracy
from tqdm import tqdm


class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, X):
        return X.view(X.shape[0], -1)


class VGG16(nn.Module):
    def __init__(self, conv_arch, fc_features, fc_hidden_units=4096, classes=1000):
        super(VGG16, self).__init__()
        self.conv_arch = conv_arch
        self.fc_features = fc_features
        self.fc_hidden_units = fc_hidden_units
        self.classes = classes
        self.net = nn.Sequential()
        for i, (num_convs, in_channels, out_channels) in enumerate(self.conv_arch):
            self.net.add_module("vgg_block_" + str(i), self.vgg_block(num_convs, in_channels, out_channels))
        self.fc = nn.Sequential(
            FlattenLayer(),
            nn.Linear(self.fc_features, self.fc_hidden_units),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.fc_hidden_units, self.fc_hidden_units),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.fc_hidden_units, self.classes)
        )
        self.net.add_module("fc", self.fc)

    def vgg_block(self, num_convs, in_channels, out_channels):
        blocks = []
        for i in range(num_convs):
            if i == 0:
                blocks.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            else:
                blocks.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
            blocks.append(nn.ReLU())
        blocks.append(nn.MaxPool2d(3, 2))
        return nn.Sequential(*blocks)

    def forward(self, X):
        return self.net(X)


if __name__ == "__main__":
    INPUT_SIZE = 224
    BATCH_SIZE = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    conv_arch = [(2, 3, 64), (2, 64, 128), (3, 128, 256), (3, 256, 512), (3, 512, 512)]
    vgg16 = VGG16(conv_arch=conv_arch, fc_features=7 * 7 * 512, fc_hidden_units=4096, classes=10)
    vgg16 = vgg16.to(device)
    optimizer = optim.Adam(vgg16.parameters(), lr=1e-3)
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
            y_pred = vgg16(X)
            loss = lossFN(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_loss += loss.cpu().item()
            sum_acc += (y_pred.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(valDL, vgg16)
        print("epoch %d: loss=%.4f \t acc=%.4f \t test acc=%.4f" % (epoch + 1, sum_loss / n, sum_acc / n, test_acc))
