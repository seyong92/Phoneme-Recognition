import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


sys.path.insert(0, '../..')


class ConvNet(nn.Module):
    """
        Audio Preprocessing :
            sr = 16000
            n_mfcc = 20

    """
    def __init__(self, N_MFCC, label_num, device):
        super(ConvNet, self).__init__()

        # model setting
        self.label_num = label_num
        self.device = device
        stride_size = int((N_MFCC - (3 - 1) - 1) / 3 + 1)

        self.cnn = nn.Sequential(
            # Layer 0 (N, 3, N_MFCC, L)
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=(3, 5), stride=1, padding=(1, 2), bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # Max Pooling (N, 128, N_MFCC, L)
            nn.MaxPool2d(kernel_size=(3, 1), stride=None, padding=0, dilation=1),
            # Layer 1 (N, 128, N_MFCC - 2, L)
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 5), stride=1, padding=(1, 2), bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # Layer 2 (N, 128, N_MFCC - 2, L)
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 5), stride=1, padding=(1, 2), bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # Layer 3 (N, 128, N_MFCC - 2, L)
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 5), stride=1, padding=(1, 2), bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # Layer 4 (N, 256, N_MFCC - 2, L)
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 5), stride=1, padding=(1, 2), bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # Layer 5 (N, 256, N_MFCC - 2, L)
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 5), stride=1, padding=(1, 2), bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # Layer 6 (N, 256, N_MFCC - 2, L)
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 5), stride=1, padding=(1, 2), bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # Layer 7 (N, 256, N_MFCC - 2, L)
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 5), stride=1, padding=(1, 2), bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # Layer 8 (N, 256, N_MFCC - 2, L)
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 5), stride=1, padding=(1, 2), bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # Layer 9 (N, 256, N_MFCC - 2, L)
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 5), stride=1, padding=(1, 2), bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(256 * stride_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, label_num),
        )

    def save(self, filepath):
        checkpoint = self.state_dict()
        filepath.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, str(filepath))

    def forward(self, x):
        x = self.cnn(x)
        x = x.transpose(1, 3).flatten(-2)
        x = self.fc(x)
        x = x.transpose(1, 2)

        return x
