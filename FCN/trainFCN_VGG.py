import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from FCN.VOC2012Dataset import VOC2012SegDataIter
from FCN.FCN_VGG_NET import VGGNet, FCNs
import time
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    batch_size = 6
    epochs = 500
    lr = 1e-4
    momentum = 0
    w_decay = 1e-5
    step_size = 50
    gamma = 0.5
    model_save_path = "./models"

    vgg11_model = VGGNet(pretrained=True, model='vgg11', requires_grad=True, remove_fc=True)
    vgg11_model = vgg11_model.to(device)
    fcn_model = FCNs(vgg11_model, n_class=21)
    fcn_model = fcn_model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.RMSprop(fcn_model.parameters(), lr=lr, momentum=momentum, weight_decay=w_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size, gamma)
    train_iter, val_iter = VOC2012SegDataIter(16, (320, 480), 2, 200)

    for epoch in tqdm(range(epochs)):
        start_t = time.time()

        for iters, (X, y) in enumerate(train_iter):
            optimizer.zero_grad()
            X = X.to(device)
            y = y.to(device)

            outputs = fcn_model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            if iters % 10 == 0:
                print("epoch{}, iter{}, loss: {}".format(epoch, iters, loss.cpu().item()))
        print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - start_t))
        scheduler.step()

    torch.save(fcn_model, model_save_path)


