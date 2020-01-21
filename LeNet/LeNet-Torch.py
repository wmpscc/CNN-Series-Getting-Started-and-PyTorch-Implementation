import torch
import torchvision
from torch import nn, optim
from utils import load_data_fashion_mnist, evaluate_accuracy
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.Sigmoid(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.Sigmoid(),
            nn.AvgPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(feature.shape[0], -1))
        return output


if __name__ == '__main__':
    net = LeNet()
    batch_size = 256
    train_iter, test_iter = load_data_fashion_mnist(batch_size, root='./dataset/FashionMNIST')
    learning_rate = 1e-3
    num_epochs = 20
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    net = net.to(device)
    print('training on ', device)
    loss = nn.CrossEntropyLoss()
    batch_count = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0., 0., 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_pred = net(X)
            l = loss(y_pred, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_pred.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time % .1f sec'
              % (epoch + 1, train_l_sum / batch_count,
                 train_acc_sum / n, test_acc, time.time() - start))
