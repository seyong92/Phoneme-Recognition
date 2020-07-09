from pprint import PrettyPrinter

import torch
import os
from phonerec.datasets import *
# from phonerec.models import *
from phonerec.trainers import Trainer

from hparams import *


pp = PrettyPrinter(indent=4)


# @dataset_config.capture
# def load_dataset(dataset_name, dataset_path, label_num, mode):
#     return globals()[dataset_name](dataset_path, label_num, mode)


# @model_config.capture
# def set_model(model_name, input_height, label_num, snapshot_path, device):
#     return globals()[model_name](input_height, label_num,
#                                  snapshot_path, device).to(device)


# @experiment_config.capture
# def set_trainer(dataset, model, log_path, snapshot_path, save_epoch, max_epoch,
#                 batch_size, learning_rate, sr, device):
#     return Trainer(dataset, model, log_path, snapshot_path, save_epoch,
#                    max_epoch, batch_size, learning_rate, sr, device)


# @experiment_config.capture
# def set_device(gpu_num):
#     if gpu_num < 0:
#         device = torch.device('cpu')
#     else:
#         device = torch.device(gpu_num)

#     return device


# @experiment_config.capture
# def call_mode(mode):
#     return mode


# @dataset_config.capture
# def call_label_num(label_num):
#     return label_num


# @experiment_config.capture
# def set_paths(exp_root, exp_name, exp_num):
#     model_path = exp_root / (exp_name + '_' + str(exp_num))
#     snapshot_path = model_path / 'snapshots'
#     hparams_path = model_path / 'hparams.json'
#     log_path = model_path / 'log'

#     if not os.path.exists(str(model_path)):
#         os.makedirs(str(model_path))
#     if not os.path.exists(str(snapshot_path)):
#         os.makedirs(str(snapshot_path))
#     if not os.path.exists(str(log_path)):
#         os.makedirs(str(log_path))

#     return model_path, snapshot_path, hparams_path, log_path


@ex_pc.automain
def main():
    # device setting
    device = set_device()

    # path setting
    model_path, snapshot_path, hparams_path, log_path = set_paths()

    # mode setting
    mode = call_mode()

    # label_num setting
    label_num = call_label_num()

    print("[EXPERIMENT]")
    pp.pprint(h_experiment())
    print("[DATASET]")
    pp.pprint(h_dataset())
    print("[MODEL]")
    pp.pprint(h_model())
    print("[PREPROCESS]")
    pp.pprint(h_preprocess())

    dataset = load_dataset(mode=mode)
    model = set_model(label_num=label_num, snapshot_path=snapshot_path,
                      device=device)

    trainer = set_trainer(dataset=dataset, model=model, log_path=log_path,
                          snapshot_path=snapshot_path, sr=16000, device=device)

    if mode == 'train':
        trainer.train()
    elif mode == 'test':
        trainer.test()
    elif mode == 'application':
        trainer.application('resources/target_audio.wav')

"""
if __name__ == '__main__':
    # config = get_arguments()
    # config = configparser.RawConfigParser()
    # config.optionxform = str
    # config.read('hparams.conf')

    pp.pprint(hparams)

    # intervals = {'log': config.log,
    #              'val': config.val,
    #              'snapshot': config.snapshot,
    #              'val_e': config.val_e,
    #              'snapshot_e': config.snapshot_e}

    dataset = globals()[dataset_name](hparams)
    model = globals()[model_name](hparams)
    model = model.to(hparams['EXPERIMENT']['DEVICE'])

    trainer = Trainer(dataset, model, hparams)

    if hparams['EXPERIMENT']['MODE'] == 'train':
        trainer.train()
    elif hparams['EXPERIMENT']['MODE'] == 'test':
        trainer.test()
    elif hparams['EXPERIMENT']['MODE'] == 'application':
        trainer.application('l-joo.wav')
"""
