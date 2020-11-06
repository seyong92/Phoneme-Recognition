from pathlib import Path

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

import wandb
from omegaconf import OmegaConf
from tqdm import tqdm

from phonerec import Timit, ConvNet, CRNN, LabelComp, eval_count


def train():
    wandb.init()
    # conf = OmegaConf.load('config-lightning.yaml')
    const = OmegaConf.create(wandb.config['CONSTANTS'])
    paths = OmegaConf.create(wandb.config['PATHS'])
    hparams = OmegaConf.create(wandb.config['HPARAMS'])
    # trainer_params = conf.trainer

    train_dataset = Timit(paths.PREPROCESSED_DATA_PATH, groups=['train'],
                          device=hparams.device, sr=const.SAMPLE_RATE,
                          num_labels=hparams.num_labels,
                          chunk_len=hparams.chunk_len)
    valid_dataset = Timit(paths.PREPROCESSED_DATA_PATH, groups=['test'],
                          device=hparams.device, sr=const.SAMPLE_RATE,
                          num_labels=hparams.num_labels)
    train_loader = DataLoader(train_dataset, hparams.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, 1, shuffle=False)

    if hparams.model_name == "ConvNet":
        model = ConvNet(const, paths, hparams).to(hparams.device)
    elif hparams.model_name == "CRNN":
        model = CRNN(const, paths, hparams).to(hparams.device)
    else:
        raise NameError('Not implemented model name.')
    label_comp = LabelComp(const.HOP_SIZE, const.N_FFT,
                           hparams.num_labels).to(hparams.device)

    criterion = CrossEntropyLoss()
    optim = Adam(model.parameters(), lr=hparams.learning_rate)
    scheduler = StepLR(optim, hparams.decay_steps, hparams.decay_rate)

    if wandb.run.resumed:
        model.load_state_dict(str(Path(wandb.run.dir) / 'model-recent.pt'))
        optim.load_state_dict(str(Path(wandb.run.dir) / 'optimizer-recent.pt'))
        scheduler.load_state_dict(str(Path(wandb.run.dir) / 'scheduler-recent.pt'))

    # trainer = Trainer(**trainer_params, logger=wandb_logger)

    valid_best_f1 = 0

    for epoch in range(hparams.max_epoch):
        for i, batch in enumerate(tqdm(train_loader, desc=f'Train epoch #{epoch}')):
            pred = model(batch['audio'])
            lbl = label_comp(batch['label'])

            #  pred_reshape (output_features, num_labels)
            pred_reshape = pred.permute(1, 0, 2).reshape(pred.size(1), -1).T

            lbl_reshape = lbl.reshape(-1)

            loss = criterion(pred_reshape, lbl_reshape)

            optim.zero_grad()
            loss.backward()
            optim.step()
            scheduler.step()

            wandb.log({'train_loss': loss.item()},
                      step=len(train_loader) * epoch + i)

        if epoch % hparams.validation_interval == 0:
            model.eval()
            with torch.no_grad():
                tp_total = torch.zeros(hparams.num_labels)
                fp_total = torch.zeros(hparams.num_labels)
                tn_total = torch.zeros(hparams.num_labels)
                fn_total = torch.zeros(hparams.num_labels)
                right_total = 0
                wrong_total = 0
                val_loss = torch.zeros(len(valid_loader))
                for j, batch in enumerate(tqdm(valid_loader, desc='Validation')):
                    pred = model(batch['audio'])
                    lbl = label_comp(batch['label'])

                    pred_reshape = pred.permute(1, 0, 2).reshape(pred.size(1), -1).T
                    # pred_reshape = torch.argmax(pred_reshape, dim=1)
                    lbl_reshape = lbl.reshape(-1)

                    tp, fp, tn, fn, r, w = eval_count(pred_reshape, lbl_reshape)

                    tp_total += tp
                    fp_total += fp
                    tn_total += tn
                    fn_total += fn
                    right_total += r
                    wrong_total += w
                    val_loss[j] = criterion(pred_reshape, lbl_reshape)

                precision = tp_total / (tp_total + fp_total + 1e-10)
                recall = tp_total / (tp_total + fn_total + 1e-10)
                accuracy = right_total / (right_total + wrong_total)
                f1_score = 2 * precision * recall / (precision + recall + 1e-10)

                wandb.log({'val_accuracy': accuracy,
                           'precision': precision.mean(),
                           'recall': recall.mean(),
                           'valid_f1': f1_score.mean(),
                           'val_loss': val_loss.mean()},
                          step=len(train_loader) * (epoch + 1) - 1)
                if valid_best_f1 < f1_score.mean():
                    valid_best_f1 = f1_score.mean()
                    torch.save(model.state_dict(),
                               str(Path(wandb.run.dir) / 'model-best.pt'))
            model.train()

        if epoch % hparams.checkpoint_interval == 0:
            torch.save(model.state_dict(),
                       str(Path(wandb.run.dir) / 'model-recent.pt'))
            torch.save(optim.state_dict(),
                       str(Path(wandb.run.dir) / 'optimizer-recent.pt'))
            torch.save(scheduler.state_dict(),
                       str(Path(wandb.run.dir) / 'scheduler-recent.pt'))


if __name__ == '__main__':
    train()
