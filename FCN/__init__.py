from tqdm import tqdm

from FCN.VOC2012Dataset import VOC2012SegDataIter
import torch
from torch import nn, optim
from torch.nn import functional as F
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


def cross_entropy2d(input, target, weight=None, size_average=True):
    # input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = input.size()
    # log_p: (n, c, h, w)
    log_p = F.log_softmax(input, dim=1)
    # log_p: (n*h*w, c)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)
    # target: (n*h*w,)
    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, weight=weight, reduction='sum')
    if size_average:
        loss /= mask.data.sum()
    return loss


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
    lossFN = nn.CrossEntropyLoss()

    num_epochs = 10
    for epoch in range(num_epochs):
        sum_loss = 0
        sum_acc = 0
        batch_count = 0
        n = 0
        for X, y in tqdm(train_iter):
            X = X.to(device)
            y = y.to(device)
            y_pred = net(X).reshape((batch_size, 21, -1))
            y = y.reshape(batch_size, -1)
            print(y_pred.shape, y.shape)
            loss = lossFN(y_pred, y)
            # loss = cross_entropy2d(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_loss += loss.cpu().item()
            sum_acc += (y_pred.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        print("epoch %d: loss=%.4f \t acc=%.4f" % (epoch + 1, sum_loss / n, sum_acc / n))
