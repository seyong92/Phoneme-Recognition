import wandb
import toml
import sys

from pprint import PrettyPrinter
from pathlib import Path

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss

from phonerec.attrs import Attrs
from phonerec import Timit, MaxOutNet

from tqdm import tqdm


pp = PrettyPrinter(indent=4)


def train():
    config = toml.load('config.toml')

    const = Attrs(config['CONSTANTS'])
    paths = Attrs(config['PATHS'])
    hparams = Attrs(config['HPARAMS'])

    if hparams.seed != -1:
        torch.manual_seed(hparams.seed)
    else:
        random_seed = torch.randint(0, sys.maxsize, (1,))[0]
        torch.manual_seed(random_seed)
        hparams.seed = random_seed

    wandb.init(config=config, resume=hparams.resume)

    train_groups, test_groups = ['train'], ['test']

    # For train dataset
    train_dataset = Timit(paths.PREPROCESSED_DATA_PATH, groups=train_groups,
                          device=hparams.device, sr=const.SAMPLE_RATE,
                          hop_size=const.HOP_SIZE, chunk_len=hparams.chunk_len)
    test_dataset = Timit(paths.PREPROCESSED_DATA_PATH, groups=test_groups,
                         device=hparams.device, sr=const.SAMPLE_RATE,
                         hop_size=const.HOP_SIZE)

    # Create model and optimizer, and load parameter if we resume experiment.
    model = MaxOutNet(hparams.input_height, hparams.num_labels,
                      hparams.device).to(hparams.device)
    optimizer = Adam(model.parameters(), hparams.learning_rate)
    if wandb.run.resumed:
        model.load_state_dict(str(Path(wandb.run.dir) / 'model-recent.pt'))
        optimizer.load_state_dict(str(Path(wandb.run.dir) / 'optimizer-recent.pt'))

    wandb.watch(model)
    scheduler = StepLR(optimizer, step_size=hparams.decay_steps,
                       gamma=hparams.decay_rate)
    loader = DataLoader(train_dataset, hparams.batch_size, shuffle=True)
    criterion = CrossEntropyLoss()

    valid_best_f1 = 0

    for epoch in range(hparams.max_epoch):
        for i, batch in enumerate(tqdm(loader, desc=f'Train epoch #{epoch}')):
            pred = model(batch['mfcc'])

            pred_reshape = pred.permute(1, 0, 2).reshape(pred.size(1), -1).T
            label_reshape = batch['label'].permute(1, 0, 2).reshape(batch['label'].size(1), -1).T
            label_reshape = torch.argmax(label_reshape, dim=1)

            loss = criterion(pred_reshape, label_reshape)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            wandb.log({'train_loss': loss.item()},
                      step=len(loader) * epoch + i)
            # writer.add_scalar('loss/train_loss', loss.item(),
            #                   global_step=len(loader) * epoch + i)

        if epoch % hparams.validation_interval == 0:
            model.eval()
            with torch.no_grad():
                tp = torch.zeros(hparams.num_labels)
                fp = torch.zeros(hparams.num_labels)
                tn = torch.zeros(hparams.num_labels)
                fn = torch.zeros(hparams.num_labels)
                right = 0
                wrong = 0
                for _, valid_data in enumerate(tqdm(test_dataset, desc='Validation')):
                    pred = model(valid_data['mfcc'].unsqueeze(0))
                    label = valid_data['label'].unsqueeze(0)

                    pred_reshape = pred.permute(1, 0, 2).reshape(pred.size(1), -1).T
                    pred_reshape = torch.argmax(pred_reshape, dim=1)
                    label_reshape = label.permute(1, 0, 2).reshape(label.size(1), -1).T
                    label_reshape = torch.argmax(label_reshape, dim=1)

                    for k in range(len(pred_reshape)):
                        if pred_reshape[k] == label_reshape[k]:
                            tp[pred_reshape[k]] += 1
                            tn += 1
                            tn[pred_reshape[k]] -= 1
                            right += 1
                        else:
                            fp[pred_reshape[k]] += 1
                            fn[label_reshape[k]] += 1

                            tn += 1
                            tn[pred_reshape[k]] -= 1
                            tn[label_reshape[k]] -= 1
                            wrong += 1

                precision = tp / (tp + fp + 1e-10)
                recall = tp / (tp + fn + 1e-10)
                accuracy = right / (right + wrong)
                f1_score = 2 * precision * recall / (precision + recall + 1e-10)
                wandb.log({'valid_accuracy': accuracy,
                           'precision': torch.mean(precision),
                           'recall': torch.mean(recall),
                           'valid_f1': torch.mean(f1_score)},
                          step=len(loader) * (epoch + 1) - 1)
                if valid_best_f1 < torch.mean(f1_score):
                    valid_best_f1 = torch.mean(f1_score)
                    torch.save(model.state_dict(),
                               str(Path(wandb.run.dir) / 'model-best.pt'))
            model.train()

        if epoch % hparams.checkpoint_interval == 0:
            torch.save(model.state_dict(),
                       str(Path(wandb.run.dir) / 'model-recent.pt'))
            torch.save(optimizer.state_dict(),
                       str(Path(wandb.run.dir) / 'optimizer-recent.pt'))


if __name__ == '__main__':
    train()
