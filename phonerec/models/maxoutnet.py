import sys

from hparams import ex_pc, model_config

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import json


sys.path.insert(0, '../..')


class MaxOutNet(nn.Module):
    """
        Audio Preprocessing :
            sr = 16000
            n_mfcc = 20

    """
    def __init__(self, input_height, label_num, snapshot_path, device):
        super(MaxOutNet, self).__init__()

        # model setting        
        self.snapshot_path = snapshot_path
        self.label_num = label_num
        self.device = device

        # (N, 3, 20, 32)
        self.conv0 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=(3, 5), stride=1, padding=(1, 3), bias=True)
        # (N, 128, 20, 34)
        self.maxpool = nn.MaxPool2d(kernel_size=(3, 1), stride=None, padding=0, dilation=1)
        # (N, 128, 18, 34)
        self.conv1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 5), stride=1, padding=(1, 2), bias=True)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 5), stride=1, padding=(1, 2), bias=True)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 5), stride=1, padding=(1, 2), bias=True)
        # (N, 256, 18, 34)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 5), stride=1, padding=(1, 2), bias=True)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 5), stride=1, padding=(1, 2), bias=True)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 5), stride=1, padding=(1, 2), bias=True)
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 5), stride=1, padding=(1, 2), bias=True)
        self.conv8 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 5), stride=1, padding=(1, 2), bias=True)
        self.conv9 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 5), stride=1, padding=(1, 2), bias=True)
        # (N, 256, 18, 34)

        self.fc1 = nn.Linear(256 * input_height * 3, 1024) # 256 * 18 * 3
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, label_num)

    def save(self, state, suffix='_model.pt'):
        fname = self.snapshot_path / (state + suffix)
        print('Saving at', fname)

        checkpoint = {'model': self.state_dict()}
        torch.save(checkpoint, str(fname))

    def forward(self, x):
        N = x.size(0)
        # input: N x 3 x 20 x 32
        x = F.relu(self.conv0(x))
        # size:  N x 128 x 20 x 34
        # x = self.maxpool(x)
        # size:  N x 128 x 18 x 34
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))

        x_merge = torch.zeros((N, self.label_num, x.size(3) - 2)).to(self.device)
        for m_idx in range(1, x_merge.size(2)):
            x_div = x[:, :, :, m_idx - 1: m_idx + 2].clone()
            x_div = x_div.view(x_div.size(0), -1)
            x_div = F.relu(self.fc1(x_div))
            x_div = F.relu(self.fc2(x_div))
            x_div = F.relu(self.fc3(x_div))
            x_div = F.log_softmax(x_div, dim=1)
            x_merge[:, :, m_idx - 1] = x_div

        return x_merge
