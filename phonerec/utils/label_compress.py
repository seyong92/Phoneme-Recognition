import torch
from torch import nn


class LabelComp(nn.Module):
    def __init__(self, hop_size, fft_size, num_labels):
        super(LabelComp, self).__init__()

        self.hop_size = hop_size
        self.fft_size = fft_size
        self.num_labels = num_labels

        self.pad = nn.ReflectionPad1d(fft_size // 2)

        self.conv_sum = nn.Conv2d(1, 1, kernel_size=(1, fft_size), stride=(1, hop_size), bias=False)
        self.conv_sum.weight = nn.Parameter(torch.ones(1, 1, 1, fft_size))
        self.conv_sum.weight.requires_grad = False

    def forward(self, lbl):
        lbl = self.pad(lbl.float().unsqueeze(1))
        lbl_one_hot = torch.zeros(lbl.size(0), self.num_labels, lbl.size(-1)).to(lbl.device)
        lbl_one_hot.scatter_(1, lbl.long(), 1)
        lbl_one_hot = lbl_one_hot.unsqueeze(1)
        lbl_comp = self.conv_sum(lbl_one_hot)
        lbl_comp = torch.argmax(lbl_comp.squeeze(1), axis=1)

        return lbl_comp
