import torch
from torch import nn, optim
from utils.load_data_Fnt10 import load_data_Fnt10
from utils import evaluate_accuracy
from tqdm import tqdm

class AlexNet(nn.Module):
    def __init__(self, classes=1000):
        super(AlexNet, self).__init__()
        self.classes = classes
        self.conv = nn.Sequential(
            nn.Conv2d(3, 96, 11, 4),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(6 * 6 * 256, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, self.classes)   # 使用交叉熵作为损失函数，无需使用Softmax
        )

    def forward(self, img):
        self.feature = self.conv(img)
        self.out = self.fc(self.feature.view(self.feature.shape[0], -1))
        return self.out


if __name__ == "__main__":
    INPUT_SIZE = 227
    BATCH_SIZE = 32

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    alexNet = AlexNet(classes=10)
    alexNet = alexNet.to(device)
    optimizer = optim.Adam(alexNet.parameters(), lr=1e-3)
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
            y_pred = alexNet(X)
            loss = lossFN(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_loss += loss.cpu().item()
            sum_acc += (y_pred.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(valDL, alexNet)
        print("epoch %d: loss=%.4f \t acc=%.4f \t test acc=%.4f"%(epoch + 1, sum_loss / n, sum_acc / n, test_acc))



