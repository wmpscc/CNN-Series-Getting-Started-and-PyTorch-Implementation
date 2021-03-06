from tqdm import tqdm

from FCN.VOC2012Dataset import VOC2012SegDataIter
import torch
from torch import nn, optim
import numpy as np
from torchvision import models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_classes = 21


def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype='float32')
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.tensor(weight)


if __name__ == '__main__':
    batch_size = 4
    train_iter, val_iter = VOC2012SegDataIter(batch_size, (320, 480), 2, 200)

    resnet18 = models.resnet18(pretrained=True)
    resnet18_modules = [layer for layer in resnet18.children()]
    net = nn.Sequential()
    for i, layer in enumerate(resnet18_modules[:-2]):
        net.add_module(str(i), layer)

    net.add_module("LinearTranspose", nn.Conv2d(512, num_classes, kernel_size=1))
    net.add_module("ConvTranspose2d",
                   nn.ConvTranspose2d(num_classes, num_classes, kernel_size=64, padding=16, stride=32))

    net[-1].weight = nn.Parameter(bilinear_kernel(num_classes, num_classes, 64), True)
    net[-2].weight = nn.init.xavier_uniform_(net[-2].weight)

    net = net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    lossFN = nn.BCEWithLogitsLoss()

    num_epochs = 10
    for epoch in range(num_epochs):
        sum_loss = 0
        sum_acc = 0
        batch_count = 0
        n = 0
        for X, y in tqdm(train_iter):
            X = X.to(device)
            y = y.to(device)
            y_pred = net(X)
            loss = lossFN(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_loss += loss.cpu().item()
            n += y.shape[0]
            batch_count += 1
        print("epoch %d: loss=%.4f" % (epoch + 1, sum_loss / n))
