import torch
from torch import nn, optim
import torchex.nn as exnn
from SSD.utils import MultiBoxPrior


class TinySSD(nn.Module):
    def __init__(self, num_classes, num_achors, sizes, ratios):
        super(TinySSD, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_achors
        self.sizes = sizes
        self.ratios = ratios
        self.out_filters = [64, 128, 128, 128, 128]
        for i in range(5):
            # 即赋值语句self.blk_i = get_blk(i)
            setattr(self, 'blk_%d' % i, self.get_block(i))
            setattr(self, 'cls_%d' % i,
                    self.cls_predictor(num_anchors=num_achors, num_classes=num_classes, in_channel=self.out_filters[i]))
            setattr(self, 'bbox_%d' % i, self.bbox_predictor(num_anchors=num_achors, in_channel=self.out_filters[i]))

    def bbox_predictor(self, num_anchors, in_channel):
        return nn.Conv2d(in_channel, num_anchors * 4, kernel_size=3, padding=1)

    def cls_predictor(self, num_anchors, num_classes, in_channel):
        return nn.Conv2d(in_channel, num_anchors * (num_classes + 1), kernel_size=3, padding=1)

    def flatten_pred(self, pred):
        in_c = pred.size(0)
        return pred.permute((0, 2, 3, 1)).reshape(in_c, -1)

    def concat_preds(self, preds):
        return torch.cat([self.flatten_pred(p) for p in preds], dim=1)

    def down_sample_block(self, in_channels, num_channels):
        block = nn.Sequential()
        for i in range(2):
            if i == 0:
                block.add_module(name="ds_conv_" + str(i),
                                 module=nn.Conv2d(in_channels, num_channels, kernel_size=3, padding=1))
            else:
                block.add_module(name="ds_conv_" + str(i),
                                 module=nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1))
            block.add_module(name="ds_bn_" + str(i), module=nn.BatchNorm2d(num_channels))
            block.add_module(name="ds_relu_" + str(i), module=nn.ReLU())
        block.add_module(name="ds_MaxPool2D", module=nn.MaxPool2d(kernel_size=2))
        return block

    def base_net(self):
        block = nn.Sequential()
        in_channel = 3
        for num_filters in [16, 32, 64]:
            block.add_module("ds_block_filter_" + str(num_filters),
                             module=self.down_sample_block(in_channel, num_filters))
            in_channel = num_filters
        return block

    def get_block(self, i):
        if i == 0:
            block = self.base_net()
        elif i == 4:
            block = exnn.GlobalMaxPool2d()
        else:
            if i == 1:
                block = self.down_sample_block(64, 128)
            else:
                block = self.down_sample_block(128, 128)
        return block

    def block_forward(self, X, block, size, ratio, cls_predictor, bbox_predictor):
        Y = block(X)
        anchors = MultiBoxPrior(Y, sizes=size, ratios=ratio)
        cls_preds = cls_predictor(Y)
        bbox_preds = bbox_predictor(Y)
        return Y, anchors, cls_preds, bbox_preds

    def forward(self, X):
        bn = X.size(0)
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            X, anchors[i], cls_preds[i], bbox_preds[i] = self.block_forward(X,
                                                                            getattr(self, 'blk_%d' % i),
                                                                            self.sizes[i],
                                                                            self.ratios[i],
                                                                            getattr(self, 'cls_%d' % i),
                                                                            getattr(self, 'bbox_%d' % i))
        anchors = torch.cat([a for a in list(anchors)], dim=1)
        cls_preds = self.concat_preds(cls_preds).reshape(bn, -1, self.num_classes + 1)
        bbox_preds = self.concat_preds(bbox_preds)
        return anchors, cls_preds, bbox_preds
