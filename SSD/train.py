import torch
from torch import nn, optim
from SSD.PikachuDetDataset import load_data_pikachu
from SSD.TinySSD import TinySSD
from SSD.utils import MultiBoxTarget

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    # print(cls_preds.shape, cls_labels.shape, bbox_preds.shape, bbox_labels.shape, bbox_masks.shape)
    cls_preds = cls_preds.to(device)
    cls_labels = cls_labels.to(device)
    bbox_preds = bbox_preds.to(device)
    bbox_labels = bbox_labels.to(device)
    bbox_masks = bbox_masks.to(device)

    cls_loss = nn.CrossEntropyLoss()
    bbox_loss = nn.L1Loss()
    cls = cls_loss(cls_preds, cls_labels)
    bbox = bbox_loss(bbox_preds * bbox_masks, bbox_labels * bbox_masks)
    return cls, bbox


def cls_eval(cls_preds, cls_labels):
    # 由于类别预测结果放在最后一维，argmax需要指定最后一维
    return (cls_preds.argmax(axis=-1) == cls_labels).sum()


def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    return ((bbox_labels - bbox_preds) * bbox_masks).abs().sum()


if __name__ == '__main__':

    sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79],
             [0.88, 0.961]]
    ratios = [[1, 2, 0.5]] * 5
    num_anchors = len(sizes[0]) + len(ratios[0]) - 1

    net = TinySSD(1, num_anchors, sizes, ratios)
    net = net.to(device)
    train_iter, val_iter = load_data_pikachu(16)
    optimizer = optim.Adam(net.parameters(), lr=1e-3)

    num_epochs = 1
    for epoch in range(num_epochs):
        acc_sum, mae_sum, n, m, it = 0.0, 0.0, 0, 0, 0
        for batch in train_iter:
            it += 1
            X = batch['image'].to(device)
            y = batch['label'].to(device)

            # 生成多尺度锚框，为每个锚框预测类别和偏移量
            anchors, cls_preds, bbox_preds = net(X)
            # 为每个锚框标注类别和偏移量
            bbox_labels, bbox_masks, cls_labels = MultiBoxTarget(anchors, y)
            # 根据类别和偏移量的预测和标注计算损失函数
            cls_preds = cls_preds.reshape(-1, 2)
            cls_labels = cls_labels.reshape(-1)
            cls_loss, bbox_loss = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks)
            l = cls_loss + bbox_loss
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            acc_sum += cls_eval(cls_preds.cpu(), cls_labels.cpu())
            n += cls_labels.size(0)

            mae_sum += bbox_eval(bbox_preds.cpu(), bbox_labels.cpu(), bbox_masks.cpu())
            m += bbox_labels.size(0)
            sum_loss = l.cpu().item()
            print("iter:", it, "cls loss:%.5f" % cls_loss.cpu().item(), "bbox loss:%.5f" % bbox_loss.cpu().item(),
                  "total loss:%.5f" % sum_loss)
        print("epoch %2d, class err %.2e, bbox mae %.2e" % (epoch + 1, 1 - acc_sum / n, mae_sum / m))
    torch.save(net.state_dict(), './ssd.pt')
