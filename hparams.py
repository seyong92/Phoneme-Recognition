from pathlib import Path
from sacred import Ingredient, Experiment


# experiment_config = Ingredient('experiment')
# dataset_config = Ingredient('dataset')
# model_config = Ingredient('model')
preprocess_config = Ingredient('preprocess')


# ex_pc = Experiment('Phoneme_Classifier',
#                    ingredients=[experiment_config,
#                                 dataset_config,
#                                 model_config])
# ex_pp = Experiment('Audio Preprocess', ingredients=[preprocess_config,
#                                                     dataset_config])


# @experiment_config.config
# def h_experiment():
#     exp_root = Path('/media/dataset/Phoneme_Recognition/sessions')
#     exp_name = 'train39'
#     exp_num = 3

#     gpu_num = 0  # -1 for CPU, others for # of GPU.

#     mode = 'train'
#     load_from = 'sessions/Train_2'
#     save_epoch = 10

#     batch_size = 32
#     learning_rate = 0.0003
#     max_epoch = 100


# @dataset_config.config
# def h_dataset():
#     dataset_name = 'Timit'
#     dataset_path = Path('/media/dataset/TIMIT_preprocessed/')

#     phoneset_61 = 'timit61.phoneset'
#     phoneset_39 = 'timit39.phoneset'

#     label_num = 39


# @model_config.config
# def h_model():
#     model_name = 'MaxOutNet'
#     input_height = 20


# @preprocess_config.config
# def h_preprocess():
#     workers = 5

#     dtype = 'mfcc'  # wav, mel, cqt, spec, mfcc
#     sr = 16000
#     hops = 512
#     n_fft = 2048
#     n_mfcc = 20
#     mels = 128

#     chunk_len = 1  # in seconds
#     original_dataset_path = Path('/media/bach1/dataset/TIMIT/LDC93S1-TIMIT/timit')
#     target_path = Path('/media/xerox_dataset/TIMIT_preprocessed/')
