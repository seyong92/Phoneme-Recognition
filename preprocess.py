import soundfile as sf
import numpy as np
from scipy.stats import mode
import torch
from tqdm import tqdm
from pathlib import Path
import yaml
from phonerec import Attrs


phn_dict = dict()


def process(wav_file, phn_file, save_data_path, data_group):
    x, _ = sf.read(str(wav_file))

    x_label = _phn2vec(x.size, phn_file)

    file_name_base = f'{wav_file.parent.stem}_{wav_file.stem}'
    save_path = Path(save_data_path) / data_group / f'{file_name_base}.pt'

    torch.save(dict(audio=torch.from_numpy(x).float(),
                    label=torch.from_numpy(x_label).long()),
               save_path)

    # print("Processing", wav_file.name + '...')


def _phn2vec(wav_length, phn_path):
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

    # phone_mat = np.zeros((num_labels, feature_length))

    # for i in range(phone_mat.shape[1]):
    #     if i * HOP_SIZE >= len(phone_seq):
    #         continue
    #     label = int(mode(phone_seq[i * HOP_SIZE: i * HOP_SIZE + N_FFT])[0][0])
    #     phone_mat[label, i] = 1

    return phone_seq


def _create_dict61(PHONESET_61):
    dict61 = dict()
    line_idx = 0
    with open(PHONESET_61, 'r') as f:
        for line in f:
            phoneme = line.strip()
            dict61[phoneme] = line_idx
            line_idx += 1

    return dict61


def _create_dict39(PHONESET_39):
    dict39 = dict()
    line_idx = 0
    with open(PHONESET_39, 'r') as f:
        for line in f:
            phonemes = line.strip().replace(',', ' ').split()
            for phoneme in phonemes:
                dict39[phoneme] = line_idx
            line_idx += 1

    return dict39


def train(data_path, save_data_path):
    phone_dir = Path(data_path) / 'timit' / 'train'
    wav_dir = Path(data_path) / 'timit_wav' / 'train'

    train_phns = list(phone_dir.glob('**/*.phn'))
    train_wavs = list(wav_dir.glob('**/*.wav'))
    train_phns.sort()
    train_wavs.sort()

    train_path = Path(save_data_path) / 'train'
    train_path.mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(len(train_phns))):
        process(train_wavs[i], train_phns[i], save_data_path, 'train')


def test(data_path, save_data_path):
    phone_dir = Path(data_path) / 'timit' / 'test2'
    wav_dir = Path(data_path) / 'timit_wav' / 'test'

    test_phns = list(phone_dir.glob('**/*.phn'))
    test_wavs = list(wav_dir.glob('**/*.wav'))
    test_phns.sort()
    test_wavs.sort()

    test_path = Path(save_data_path) / 'test'
    test_path.mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(len(test_phns))):
        process(test_wavs[i], test_phns[i], save_data_path, 'test')


if __name__ == '__main__':
    with open('config-defaults.yaml') as yml:
        config = yaml.load(yml, Loader=yaml.FullLoader)
        paths = Attrs(config['PATHS']['value'])
        hparams = Attrs(config['HPARAMS']['value'])

    if hparams.num_labels == 61:
        phn_dict = _create_dict61(paths.PHONESET_61)
    elif hparams.num_labels == 39:
        phn_dict = _create_dict39(paths.PHONESET_39)

    train(paths.TIMIT_PATH, paths.PREPROCESSED_DATA_PATH)
    test(paths.TIMIT_PATH, paths.PREPROCESSED_DATA_PATH)
