from pathlib import Path
import torch
# from .constants import SAMPLE_RATE, HOP_SIZE
from torch.utils.data import Dataset
from tqdm import tqdm


class Timit(Dataset):
    def __init__(self, data_path, groups, device, sr, hop_size, chunk_len=None):
        self.groups = groups
        self.device = device

        self.data_path = data_path
        self.chunk_len = chunk_len

        self.sr = sr
        self.hop_size = hop_size

        self.data = []

        # for fname in train_fnames:
        #     if np.random.rand() < 0.75:
        #         self._names['train'].append(fname)
        #     else:
        #         self._names['valid'].append(fname)
        # for fname in test_fnames:
        #     self._names['test'].append(fname)

        for group in self.groups:
            for input_file in tqdm(self.files(group),
                                   desc=f'Loading group {group}'):
                data = self.load(input_file)
                if chunk_len is not None:  # protection code for short sample. should fix later
                    num_frames = int((self.sr // self.hop_size) * self.chunk_len)
                    if data['mfcc'].shape[-1] - num_frames <= 0:
                        continue
                self.data.append(data)

        print('TIMIT Dataset Loaded')

    def files(self, group):
        print('group is', group)
        files = list((Path(self.data_path) / group).glob('*.pt'))

        return files

    def load(self, file_path):
        return torch.load(str(file_path))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        result = dict()

        if self.chunk_len is not None:
            len_data = data['mfcc'].shape[-1]

            num_frames = int((self.sr // self.hop_size) * self.chunk_len)
            if (len_data - num_frames) <= 0:
                step_begin = 0
            else:
                step_begin = int(torch.randint(0, len_data - num_frames, (1,)))
            step_end = step_begin + num_frames

            result['mfcc'] = data['mfcc'][:, :, step_begin: step_end].to(self.device)
            result['label'] = data['label'][:, step_begin: step_end].to(self.device)
        else:
            result['mfcc'] = data['mfcc'].to(self.device)
            result['label'] = data['label'].to(self.device)

        return result
