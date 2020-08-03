import soundfile as sf
import numpy as np
from scipy.stats import mode
import torch
import librosa
from tqdm import tqdm
import re
from pathlib import Path

from phonerec.constants import DTYPE, HOP_SIZE, N_FFT, N_MFCC, N_MELS
from phonerec.paths import TIMIT_PATH, PREPROCESSED_DATA_PATH, PHONESET_61, PHONESET_39
from hparams import ex


@ex.capture
def process(wav_file, phn_file, data_group):
    x, sr = sf.read(str(wav_file))

    # feature selection
    if DTYPE == 'mfcc':
        x_feature = librosa.feature.mfcc(x, sr, n_mfcc=N_MFCC, n_mels=N_MELS,
                                         n_fft=N_FFT, hop_length=HOP_SIZE)
        x_feature_delta = librosa.feature.delta(x_feature)
        x_feature_delta2 = librosa.feature.delta(x_feature, order=2)

        x_feature = np.expand_dims(x_feature, 0)
        x_feature_delta = np.expand_dims(x_feature_delta, 0)
        x_feature_delta2 = np.expand_dims(x_feature_delta2, 0)

        x_feature_cat = np.concatenate((x_feature, x_feature_delta,
                                        x_feature_delta2), 0)

    x_label = _phn2vec(x.size, x_feature.shape[-1], phn_file)

    file_name_base = f"{wav_file.parent.stem}_{wav_file.stem}"
    save_path = Path(PREPROCESSED_DATA_PATH) / data_group / f'{file_name_base}.pt'

    torch.save(dict(mfcc=torch.from_numpy(x_feature_cat),
                    label=torch.from_numpy(x_label),
                    path=save_path),
               save_path)

    # print("Processing", wav_file.name + '...')


@ex.capture
def train():
    phone_dir = Path(TIMIT_PATH) / 'timit' / 'train'
    wav_dir = Path(TIMIT_PATH) / 'timit_wav' / 'train'

    train_phns = list(phone_dir.glob('**/*.phn'))
    train_wavs = list(wav_dir.glob('**/*.wav'))
    train_phns.sort()
    train_wavs.sort()

    train_path = Path(PREPROCESSED_DATA_PATH) / 'train'
    train_path.mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(len(train_phns))):
        process(train_wavs[i], train_phns[i], 'train')


@ex.capture
def test():
    phone_dir = Path(TIMIT_PATH) / 'timit' / 'test2'
    wav_dir = Path(TIMIT_PATH) / 'timit_wav' / 'test'

    test_phns = list(phone_dir.glob('**/*.phn'))
    test_wavs = list(wav_dir.glob('**/*.wav'))
    test_phns.sort()
    test_wavs.sort()

    test_path = Path(PREPROCESSED_DATA_PATH) / 'test'
    test_path.mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(len(test_phns))):
        process(test_wavs[i], test_phns[i], 'test')


@ex.capture
def _phn2vec(wav_length, feature_length, phn_path, num_labels):
    global phn_dict
    phone_seq = np.zeros(wav_length)
    with open(str(phn_path)) as f:
        for line in f:
            sample_start, sample_end, label_char = line.split(' ')
            sample_start = int(sample_start)
            sample_end = int(sample_end)
            label_char = label_char.strip().replace('-', '')
            if label_char == 'q':
                continue
            label_num = phn_dict[label_char]

            phone_seq[sample_start: sample_end] = label_num

    phone_mat = np.zeros((num_labels, feature_length))

    for i in range(phone_mat.shape[1]):
        if i * HOP_SIZE >= len(phone_seq):
            continue
        label = int(mode(phone_seq[i * HOP_SIZE: i * HOP_SIZE + N_FFT])[0][0])
        phone_mat[label, i] = 1

    return phone_mat


def _create_dict61():
    dict61 = dict()
    line_idx = 0
    with open(str(PHONESET_61), 'r') as f:
        for line in f:
            phoneme = line.strip()
            dict61[phoneme] = line_idx
            line_idx += 1

    return dict61


def _create_dict39():
    dict39 = dict()
    line_idx = 0
    with open(str(PHONESET_39), 'r') as f:
        for line in f:
            phonemes = line.strip().replace(',', ' ').split()
            for phoneme in phonemes:
                dict39[phoneme] = line_idx
            line_idx += 1

    return dict39


phn_dict = dict()


@ex.automain
def main(num_labels):
    global phn_dict
    if num_labels == 61:
        phn_dict = _create_dict61()
    elif num_labels == 39:
        phn_dict = _create_dict39()

    train()
    test()
