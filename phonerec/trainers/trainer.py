import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
import librosa
from pathlib import Path
from .. import utils as U
from tqdm import tqdm


sys.path.insert(0, '../..')


class Trainer:
    def __init__(self, dataset, model, log_path, snapshot_path, save_epoch,
                 max_epoch, batch_size, learning_rate, sr, device):
        self.model = model
        self.dataset = dataset
        self.save_epoch = save_epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.sr = sr

        self.device = device

        if self.device.type == 'cpu':
            self.use_gpu = False
        else:
            self.use_gpu = True

        # setup directories
        self.log_path = log_path
        self.snapshot_path = snapshot_path

        # TODO: Select optimizer by hparams
        self.optimizer = Adam(params=model.parameters(), lr=self.learning_rate)

        self.data_per_epoch = len(dataset)

        # self.dtype = hparams['PREPROCESS']['DTYPE']
        # self.n_fft = hparams['PREPROCESS']['N_FFT']
        # self.hops = hparams['PREPROCESS']['HOPS']
        # self.n_mfcc = hparams['PREPROCESS']['N_MFCC']
        # self.sr = hparams['PREPROCESS']['SR']

        self.status = {'step': 0, 'epoch': 0, 'epoch_total': max_epoch,
                       'iter': 0, 'iter_total': 0}
        # self.scalar = ['loss']
        self.best_val = 999999

        # self.log_path = model.log_path

        self.loss = nn.KLDivLoss(reduction='batchmean')

    def log(self):
        with open(str(self.log_path), 'a') as f:
            trackers = list((s, self.status[s]) for s in self.scalar)
            # f.write(now(self.mode, self.status['epoch'], self.status['step'], *trackers) + '\n')

    def infer_batch(self, data):
        def wrap_output(y):
            y = y.type(torch.FloatTensor)
            y = y.to(self.device)
            return y

        x, y = data
        x = x.to(self.device)
        y = wrap_output(y)
        y_ = self.model(x)
        return y_, y

    def get_loss(self, y_, y):
        loss = self.loss(y_, y)
        self.status['loss'] = loss.item()
        self.status['accuracy'] = U.accuracy(y_, y)
        self.status['f1'] = U.f1(y_, y)

        return loss

    def train(self):
        self.to_train()
        dataloader = DataLoader(self.dataset,
                                batch_size=self.batch_size,
                                shuffle=True,
                                num_workers=4,
                                pin_memory=self.use_gpu)

        # For each epoch.
        for e in range(self.status['epoch_total']):
            self.status['iter_total'] = len(dataloader)
            self.status['epoch'] = e
            print('Training epoch', '#' + str(e), 'Total', self.status['iter_total'], 'batches')

            # For each batch.
            data_tqdm = tqdm(dataloader, ncols=100)
            data_tqdm.set_description("Processing the epoch %d" %
                                      self.status['epoch'])

            for b_id, data in enumerate(data_tqdm):
                y_, y = self.infer_batch(data)

                # compute loss and store in status, perform backprop
                self.optimizer.zero_grad()
                loss = self.get_loss(y_, y)
                loss.backward()
                self.optimizer.step()

                self.status['iter'] = b_id
                # self.status['step'] += 1

                # compute mean loss over log interval and log
                # self.log()

                # print('Epoch %d: %.4f, Accuracy: %.4f' % (e, e / self.status['iter_total'], self.status['accuracy']))

            self.validate()
            if self.status['epoch'] % 10 == 0:
                self.save_state('recent')
            self.to_train()

        print('[Training Finished] Epochs:', e)
        self.save_state('recent')

    def validate(self):
        self.to_valid()

        dataloader = DataLoader(self.dataset,
                                batch_size=self.batch_size,
                                shuffle=False,
                                num_workers=4,
                                pin_memory=self.use_gpu)

        n_batch = len(dataloader)
        epoch_loss = 0
        epoch_acc = 0
        with torch.no_grad():
            for data in dataloader:
                y_, y = self.infer_batch(data)
                self.get_loss(y_, y)
                epoch_loss += self.status['loss'] * y.size(0)
                epoch_acc += self.status['accuracy'] * y.size(0)
                epoch_f1 += self.status['f1'] * y.size(0)

            epoch_loss /= len(dataloader.dataset)
            epoch_acc /= len(dataloader.dataset)
            epoch_f1 /= len(dataloader.dataset)


        # for s in self.scalar:
        #     self.status[s] = self.status['epoch_' + s] / n_batch
        # self.log()

        val_loss = epoch_loss

        print(self.best_val, '->', val_loss, 'diff:', self.best_val - val_loss,
              'accuracy:', epoch_acc, epoch_f1)
        if val_loss < self.best_val:
            self.best_val = val_loss
            self.save_state('best')
            # if self.status['epoch'] % 10 == 0:
            #     self.save_state('best')

    def test(self):
        self.model.load_state_dict(torch.load(self.snapshot_path / 'best_model.pt', map_location=self.device)['model'])
        self.optimizer.load_state_dict(torch.load(self.snapshot_path / 'best_opt.pt', map_location=self.device)['optimizer'])

        self.to_test()
        dataloader = DataLoader(self.dataset,
                                batch_size=self.batch_size,
                                shuffle=False,
                                num_workers=4,
                                pin_memory=self.use_gpu)
        print(len(dataloader), 'batches')

        epoch_loss = 0
        epoch_acc = 0
        with torch.no_grad():
            data_tqdm = tqdm(dataloader, ncols=100)
            data_tqdm.set_description("Test set evaluation")

            for b_id, data in enumerate(data_tqdm):
                y_, y = self.infer_batch(data)
                self.get_loss(y_, y)
                epoch_loss += self.status['loss'] * y.size(0)
                epoch_acc += self.status['accuracy'] * y.size(0)
                epoch_f1 += self.status['f1'] * y.size(0)

            epoch_loss /= len(dataloader.dataset)
            epoch_acc /= len(dataloader.dataset)
            epoch_f1 /= len(dataloader.dataset)

        print("test accuracy:", epoch_acc, "test f1:", epoch_f1)


    def application(self, wav_file):
        import numpy as np
        import librosa


        self.model.load_state_dict(torch.load(self.snapshot_path / 'best_model.pt', map_location=self.device)['model'])
        self.optimizer.load_state_dict(torch.load(self.snapshot_path / 'best_opt.pt', map_location=self.device)['optimizer'])
        self.model.eval()


        x, sr = librosa.core.load(str(wav_file), sr=self.sr)

        # feature selection
        # if self.dtype == 'mfcc':
        x_feature = librosa.feature.mfcc(x, sr, n_mfcc=20, n_fft=2048, hop_length=512)
        x_feature_delta = librosa.feature.delta(x_feature)
        x_feature_delta2 = librosa.feature.delta(x_feature, order=2)

        x_feature = np.expand_dims(x_feature, 0)
        x_feature_delta = np.expand_dims(x_feature_delta, 0)
        x_feature_delta2 = np.expand_dims(x_feature_delta2, 0)

        x_feature = np.concatenate((x_feature, x_feature_delta, x_feature_delta2),0)

        x_feature_tensor = torch.from_numpy(x_feature).type(torch.FloatTensor).to(self.device)
        x_feature_tensor = x_feature_tensor.unsqueeze(0)

        with torch.no_grad():
            y_ = self.model(x_feature_tensor)

            y_np = np.array(y_.cpu()).squeeze()
            np.save('resources/' + Path(wav_file).stem + '.npy', y_np)

    def to_train(self):
        self.mode = 'train'
        self.model.train()
        self.dataset.mode = 'train'

    def to_valid(self):
        self.mode = 'valid'
        self.model.eval()
        self.dataset.mode = 'valid'

    def to_test(self):
        self.mode = 'test'
        self.model.eval()
        self.dataset.mode = 'test'

    def save_optimizer(self, state, suffix='_opt.pt'):
        # name = [best, recent, epoch_step].pt
        fname = self.snapshot_path / (state + suffix)
        checkpoint = {'optimizer': self.optimizer.state_dict()}
        torch.save(checkpoint, str(fname))

    def save_state(self, state):
        print('Saving at', state)
        self.model.save(state)
        self.save_optimizer(state)

    # def load_state(self, state):
    #     weights = U.get_checkpoint(request, 'model', nsml_dir=nsml_dir)
    #     self.model.load_state_dict(weights)

    #     optimizer_state = U.get_checkpoint(request, 'optimizer', nsml_dir=nsml_dir)
    #     self.optimizer.load_state_dict(optimizer_state)