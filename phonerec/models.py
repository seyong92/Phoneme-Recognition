import sys
from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from nnAudio import Spectrogram as Spec
import pytorch_lightning as pl

from .utils import LabelComp, eval_count
from .datasets import Timit


sys.path.insert(0, '../..')


class PhonemeClassifier(nn.Module):
    def __init__(self, const, paths, hparams):
        super(PhonemeClassifier, self).__init__()

        self.device = hparams.device

        if const.DTYPE == 'mfcc':
            self.spec = Spec.MFCC(const.SAMPLE_RATE, const.N_MFCC,
                                  n_fft=const.N_FFT, n_mels=const.N_MELS,
                                  hop_length=const.HOP_SIZE,
                                  device=self.device)
            self.feature_size = const.N_MFCC
        elif const.DTYPE == 'melspec':
            self.spec = Spec.Melspectrogram(const.SAMPLE_RATE, const.N_FFT,
                                            const.N_FFT, const.N_MELS,
                                            const.HOP_SIZE,
                                            device=self.device)
            self.feature_size = const.N_MELS
        else:
            raise NameError

        self.label_comp = LabelComp(const.HOP_SIZE, const.N_FFT,
                                    hparams.num_labels).to(self.device)

        self.paths = paths
        self.const = const
        self.hparams = hparams

    @abstractmethod
    def forward(self):
        pass


class ConvNet(PhonemeClassifier):
    def __init__(self, const, paths, hparams):
        super(ConvNet, self).__init__(const, paths, hparams)

        # model setting
        stride_size = int((self.feature_size - (3 - 1) - 1) / 3 + 1)

        self.cnn = nn.Sequential(
            # Layer 0 (N, 1, N_MFCC, L)
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(3, 5),
                      stride=1, padding=(1, 2), bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 1), stride=None, padding=0, dilation=1),
            # Layer 1 (N, 128, stride_size, L)
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 5),
                      stride=1, padding=(1, 2), bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # Layer 2 (N, 128, stride_size, L)
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 5),
                      stride=1, padding=(1, 2), bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # Layer 3 (N, 128, stride_size, L)
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 5),
                      stride=1, padding=(1, 2), bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # Layer 4 (N, 256, stride_size, L)
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 5),
                      stride=1, padding=(1, 2), bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # Layer 5 (N, 256, stride_size, L)
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 5),
                      stride=1, padding=(1, 2), bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # Layer 6 (N, 256, stride_size, L)
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 5),
                      stride=1, padding=(1, 2), bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # Layer 7 (N, 256, stride_size, L)
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 5),
                      stride=1, padding=(1, 2), bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # Layer 8 (N, 256, stride_size, L)
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 5),
                      stride=1, padding=(1, 2), bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # Layer 9 (N, 256, stride_size, L)
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 5),
                      stride=1, padding=(1, 2), bias=True),
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
            nn.Linear(1024, hparams.num_labels),
        )

    def save(self, filepath):
        checkpoint = self.state_dict()
        filepath.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, str(filepath))

    def forward(self, x):
        x = self.spec(x)
        x = x.unsqueeze(1)
        x = self.cnn(x)
        x = x.transpose(1, 3).flatten(-2)
        x = self.fc(x)
        x = x.transpose(1, 2)

        return x  # (n_batch, num_labels, time_stamp)


class ConvStack(nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()

        # input is batch_size * 1 channel * frames * input_features
        # Layer 0 (N, 1, N_MFCC, L)
        self.cnn = nn.Sequential(
            # layer 0
            nn.Conv2d(1, output_features // 16, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 16),
            nn.ReLU(),
            # layer 1
            nn.Conv2d(output_features // 16, output_features // 16, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 16),
            nn.ReLU(),
            # layer 2
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
            nn.Conv2d(output_features // 16, output_features // 8, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 8),
            nn.ReLU(),
            # layer 3
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
        )
        self.fc = nn.Sequential(
            nn.Linear((output_features // 8) * (input_features // 4), output_features),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = x.permute(0, 1, 3, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2).flatten(-2)
        x = self.fc(x)
        return x


class BiLSTM(nn.Module):
    inference_chunk_length = 512

    def __init__(self, input_features, recurrent_features):
        super().__init__()
        self.rnn = nn.LSTM(input_features, recurrent_features, batch_first=True, bidirectional=True)

    def forward(self, x):
        if self.training:
            return self.rnn(x)[0]
        else:
            # evaluation mode: support for longer sequences that do not fit in memory
            batch_size, sequence_length, input_features = x.shape
            hidden_size = self.rnn.hidden_size
            num_directions = 2 if self.rnn.bidirectional else 1

            h = torch.zeros(num_directions, batch_size, hidden_size, device=x.device)
            c = torch.zeros(num_directions, batch_size, hidden_size, device=x.device)
            output = torch.zeros(batch_size, sequence_length, num_directions * hidden_size, device=x.device)

            # forward direction
            slices = range(0, sequence_length, self.inference_chunk_length)
            for start in slices:
                end = start + self.inference_chunk_length
                output[:, start:end, :], (h, c) = self.rnn(x[:, start:end, :], (h, c))

            # reverse direction
            if self.rnn.bidirectional:
                h.zero_()
                c.zero_()

                for start in reversed(slices):
                    end = start + self.inference_chunk_length
                    result, (h, c) = self.rnn(x[:, start:end, :], (h, c))
                    output[:, start:end, hidden_size:] = result[:, :, hidden_size:]

            return output


class CRNN(PhonemeClassifier):
    def __init__(self, const, paths, hparams, model_complexity=48):
        super(CRNN, self).__init__(const, paths, hparams)

        model_size = model_complexity * 16
        sequence_model = lambda input_size, output_size: BiLSTM(input_size, output_size // 2)

        self.conv_stack = nn.Sequential(
            ConvStack(self.feature_size, model_size),
            sequence_model(model_size, model_size),
            nn.Linear(model_size, hparams.num_labels),
        )

    def forward(self, x):
        x = self.spec(x)
        x = self.conv_stack(x)
        x = x.permute(0, 2, 1)

        return x
