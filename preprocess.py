import librosa
import numpy as np
import shutil
from pathlib import Path
from multiprocessing import Pool
import re

from hparams import preprocess_config


def multi(args):
    with Pool(self.workers) as p:
        p.starmap(self.process, args)


@preprocess_config.capture
def process(wav_file, save_path, dtype, sr, n_mfcc, n_fft, hops, chunk_len):
    x, sr = librosa.core.load(str(wav_file), sr=sr)

    # feature selection
    if dtype == 'mfcc':
        x_feature = librosa.feature.mfcc(x, sr, n_mfcc=n_mfcc,
                                         n_fft=n_fft, hop_length=hops)
        x_feature_delta = librosa.feature.delta(x_feature)
        x_feature_delta2 = librosa.feature.delta(x_feature, order=2)

        x_feature = np.expand_dims(x_feature, 0)
        x_feature_delta = np.expand_dims(x_feature_delta, 0)
        x_feature_delta2 = np.expand_dims(x_feature_delta2, 0)

        x_feature = np.concatenate((x_feature, x_feature_delta,
                                    x_feature_delta2), 0)

    x_label = _phn2vec(wav_file.parent /
                       Path(str(wav_file.stem) + '.phn'), hops)

    if x_label.shape[1] < x_feature.shape[2]:
        x_label = np.append(x_label,
                            np.zeros((61,
                                      x_feature.shape[2] - x_label.shape[1])),
                            axis=1)
    elif x_label.shape[1] > x_feature.shape[2]:
        x_feature = np.append(x_feature,
                              np.zeros((3, n_mfcc,
                                        x_label.shape[1] - x_feature.shape[2])
                                       ), axis=2)

    time_ptr = 0
    file_label = 0
    feature_hop = int(np.ceil(chunk_len * sr / hops))

    rest_zeros = feature_hop - x_feature.shape[1] % feature_hop
    x_feature = np.append(x_feature, np.zeros((3, n_mfcc, rest_zeros)), axis=2)
    x_label_sil = np.zeros((61, rest_zeros))
    x_label_sil[dict61['h#'], :] = 1
    x_label = np.append(x_label, x_label_sil, axis=1)

    while time_ptr + feature_hop < x_label.shape[1]:
        np.save(save_path /
                (wav_file.parent.stem + '_' + wav_file.stem + '_' +
                 str(file_label) + '.npy'),
                (x_feature[:, :, time_ptr: time_ptr + feature_hop],
                 x_label[:, time_ptr: time_ptr + feature_hop]))
        file_label += 1
        time_ptr += feature_hop

    print("Processing", wav_file.name + '...')


@preprocess_config.capture
def train(original_dataset_path, target_path):
    train_source = original_dataset_path / 'train'
    train_wavs = train_source.glob('**/*.wav')
    train_phns = train_source.glob('**/*.phn')
    target_path = target_path / 'train'
    target_path.mkdir(parents=True, exist_ok=True)

    args = [(wav, target_path) for cnt, wav in enumerate(train_wavs)]
    multi(args)


@preprocess_config.capture
def test(dataset_path, target_path):
    test_source = dataset_path / 'test2'
    test_wavs = test_source.glob('**/*.wav')
    test_phns = test_source.glob('**/*.phn')
    target_path = target_path / 'test'
    target_path.mkdir(parents=True, exist_ok=True)

    args = [(wav, target_path) for cnt, wav in enumerate(test_wavs)]
    self.multi(args)


@dataset_config.capture
def _create_dict61(phoneset_61):
    dict61 = dict()
    line_idx = 0
    with open(str(phoneset_61), 'r') as f:
        for line in f:
            phoneme = line.strip()
            self.dict61[phoneme] = line_idx
            line_idx += 1

    return dict61


def _phn2vec(phn, hops):
    phone_vec = np.zeros((61, 1))
    hop_ptr = hops
    with open(str(phn)) as f:
        for line in f:
            _, end, label_char = line.split(' ')
            end = int(end)
            label_num = dict61[label_char.strip().replace('-', '')]
            while hop_ptr <= end:
                label_vec = np.zeros((61, 1))
                label_vec[label_num] = 1
                phone_vec = np.append(phone_vec, label_vec, axis=1)
                hop_ptr += hops

    return phone_vec


def _make61to39mat(phone39_path):
    with open(str(phone39_path), 'r') as f:
        line_idx = 0
        for line in f:
            phoneme_list = re.split('\s+', line)[1].split(',')
            for _, phoneme in enumerate(phoneme_list):
                _61to39mat[line_idx, dict61[phoneme]] = 1
            line_idx += 1


def _vec61to39(self, vec61):
    return np.matmul(_61to39mat, vec61)

class Preprocessor:
    def __init__(self, workers, ):
        # self.target_dir = Path('_'.join(['data/IRMAS', self.dtype, str(self.sr)]))
        self.target_path = Path(config['PREPROCESS']['TARGET_PATH'])
        self.dataset_path = Path(config['PREPROCESS']['ORIGINAL_DATASET_PATH'])

        self.phone61_path = Path(config['DATASET']['PHONESET_61'])
        self.phone39_path = Path(config['DATASET']['PHONESET_39'])

        self.chunk_len = int(config['PREPROCESS']['CHUNK_LEN'])

        self.dict61 = dict()
        line_idx = 0
        with open(str(self.phone61_path), 'r') as f:
            for line in f:
                phoneme = line.strip()
                self.dict61[phoneme] = line_idx
                line_idx += 1

        self._61to39mat = np.zeros((39, 61))
        self._make61to39mat()


@ex_pp.automain
def main():
    dict61 = _create_dict61()


if __name__ == '__main__':
    import configparser


    config = configparser.ConfigParser()
    config.read('hparams.conf')

    p = Preprocessor(config)
    p.train()
    p.test()
