import torch
from torch import nn, optim
from torch.nn import functional as F
from tqdm import tqdm
from SSD.PikachuDetDataset import load_data_pikachu
from SSD.TinySSD import TinySSD
from SSD.utils import MultiBoxPrior, MultiBoxTarget

if __name__ == '__main__':
    train_iter, val_iter = load_data_pikachu(16)

    for batch in train_iter:
        print(batch['label'].shape, batch['image'].shape)
