import torch
import torch.nn as nn
import numpy as np

class NoiseNet(nn.Module):
    def __init__(self):
        super(NoiseNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.conv_seq = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.ReLU(),
        )
        self.fc_seq = nn.Sequential(
            nn.Linear(64 * 4 ** 2, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

        self.crop_size = 128

    def forward(self, in_x):
        h, w = in_x.shape[2:]
        x, y = np.random.randint(0, w - self.crop_size), np.random.randint(0, h - self.crop_size)

        new_in_x = in_x[:, :, y:y + self.crop_size, x:x + self.crop_size]
        x = self.conv1(new_in_x)
        x = self.conv_seq(x)

        x = nn.Flatten()(x)
        x = self.fc_seq(x)

        return x

    def init_weights(self):
        for m in self.parameters():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
            elif isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
