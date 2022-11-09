import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.optim.lr_scheduler import StepLR

from models import NoiseNet
from utils.DataUtils.NoiseDataset import GenImageDataset


def main():
    torch.random.manual_seed(42)

    model = NoiseNet().cuda()
    model.init_weights()

    train_loader = GenImageDataset(
        '../data/',
        phase='train',
        crop_size=128
    )

    # model.load_state_dict(torch.load('latest.pth'))

    train_loader = torch.utils.data.DataLoader(
        train_loader,
        batch_size=16, shuffle=True,
        num_workers=1, pin_memory=True, drop_last=True)

    lr = 1e-2

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=0.9)
    scheduler = StepLR(optimizer, step_size=20, gamma=.1)

    epochs = 2000
    is_train = True

    model.train()
    if not is_train:
        print("TEST!!!!!!!")
        model.eval()
        epochs = 1
        model.load_state_dict(torch.load('latest.pth'))

    for e in range(epochs):
        e_loss = 0
        acc = 0
        acc_2 = 0

        print('lr: {:.2e}'.format(optimizer.param_groups[0]['lr']))
        for idx, (img, n_img, trg) in enumerate(train_loader):
            optimizer.zero_grad()
            out, out_c = model(n_img.cuda())

            if is_train:
                loss = criterion(out, trg.cuda()) + nn.L1Loss()(out_c, img.cuda())/2
                loss.backward()
                optimizer.step()

                e_loss += loss.item()
            acc += (out.argmax(1).detach().cpu().numpy() == trg.detach().cpu().numpy()).mean()
            acc_2 += (np.abs(out.argmax(1).detach().cpu().numpy() - trg.detach().cpu().numpy()) < 2).mean()
        if idx > 0:
            idx += 1
            print(out.argmax(1).detach().cpu().numpy())
            print(trg.detach().cpu().numpy())

            print("Epoch: {}\t Loss: {:.3f}\t acc: {:.3f} \t acc_2: {:.3f} ".format(e, e_loss / idx, acc / idx,
                                                                                    acc_2 / idx), end='')
        if is_train:
            scheduler.step()
            torch.save(model.state_dict(), 'latest.pth')


if __name__ == '__main__':
    for f in os.listdir('tmp'):
        os.remove('tmp/{}'.format(f))
    main()
