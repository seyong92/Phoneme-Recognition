import numpy as np
import librosa
from pathlib import Path
import csv
import torch
import torch.utils.data
# from .. import utils as U


class Timit(torch.utils.data.Dataset):
    def __init__(self, data_path, label_num, exp_mode):

        self._names = {'train': [], 'valid': [], 'test': []}

        self.mode = 'train'

        train_fnames = (data_path / 'train').glob('*.npy')
        test_fnames = (data_path / 'test').glob('*.npy')
        for fname in train_fnames:
            if np.random.rand() < 0.75:
                self._names['train'].append(fname)
            else:
                self._names['valid'].append(fname)
        for fname in test_fnames:
            self._names['test'].append(fname)

        self._pfuncs = {}
        self.preprocess_setup()

        print('TIMIT Dataset Loaded')

    @property
    def pfuncs(self):
        return self._pfuncs[self.mode]

    def __len__(self):
        return len(self.names)

    @property
    def names(self):
        return self._names[self.mode]

    def preprocess_setup(self):
        self._pfuncs['train'] = []
        # self._pfuncs['train'] += [U.normalize()]
        self._pfuncs['valid'] = self._pfuncs['train']

        self._pfuncs['test'] = []
        # self._pfuncs['test'] += [U.normalize()]

    def preprocess(self, w, cid=-1):
        for f in self.pfuncs:
            w = f(w)
        return w if cid < 0 else w[cid]

    def get_example(self, fname):
        cid = -1
        x, y = np.load(fname, allow_pickle=True)
        x = x.astype(np.float32)
        x = self.preprocess(x, int(cid))

        y = y.astype(np.float32)

        return x, y

    def __getitem__(self, idx):
        name = self.names[idx]
        x, y = self.get_example(name)

        return x.astype(np.float32), y.astype(np.float32)
