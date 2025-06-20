from pathlib import Path
from scripts.data_handler import get_loader, load_datasets
from scripts.utils import load_yaml
from scripts.log import LogReconstructionsCallback
from scripts import constants
from models.icvae import ICVAE
from torch.cuda import is_available
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch import Trainer, seed_everything
import wandb
import argparse


def train(config, train_data, val_data, batch_size, epochs, precision, log_interval, device, workers,
          no_sync, save_path):
    seed_everything(42, workers=True)
    train_loader = get_loader(train_data, batch_size, shuffle=True, num_workers=workers)
    val_loader = get_loader(val_data, batch_size, shuffle=False, num_workers=workers)
    model = ICVAE(**config)
    run_name = f'{save_path.parent.name}_{save_path.name}'
    wandb_logger = WandbLogger(name=run_name, version=run_name, project='BrainVAE', offline=no_sync)
    checkpoint = ModelCheckpoint(dirpath=save_path, filename='{epoch:03d}', every_n_epochs=epochs // 5, save_top_k=-1,
                                 save_last=True)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    reconstruction = LogReconstructionsCallback(sample_size=8)
    trainer = Trainer(max_epochs=epochs,
                      accelerator=device,
                      precision=precision,
                      logger=wandb_logger,
                      callbacks=[checkpoint, reconstruction, lr_monitor],
                      log_every_n_steps=min(log_interval, len(train_loader) // 10)
                      )
    checkpoints = list(save_path.glob('last.ckpt'))
    wandb_logger.watch(model)
    trainer.fit(model, train_loader, val_loader, ckpt_path=checkpoints[-1] if checkpoints else None)
    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='general', help='dataset name')
    parser.add_argument('--cfg', type=str, default='age_agnostic', help='config file')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--precision', type=str, default='32', help='precision (16-mixed or 32)')
    parser.add_argument('--log_interval', type=int, default=50, help='log interval')
    parser.add_argument('--sample_size', type=int, default=2000, help='number of samples to use in smaller datasets')
    parser.add_argument('--val_size', type=float, default=0.1, help='validation size')
    parser.add_argument('--test_size', type=float, default=0.15, help='test size')
    parser.add_argument('--splits_path', type=str, default='splits', help='path to the data splits')
    parser.add_argument('--redo_splits', action='store_true', help='redo train/val/test splits')
    parser.add_argument('--final', action='store_true', help='train on the entire dataset (train + val)')
    parser.add_argument('--no_sync', action='store_true', help='do not sync to wandb')
    parser.add_argument('--device', type=str, default='gpu', help='device (gpu or cpu)')
    parser.add_argument('--workers', type=int, default=12, help='number of workers for data loading')
    parser.add_argument('--run_name', type=str, default='', help='(optional) add prefix to default run name')

    args = parser.parse_args()
    config = load_yaml(Path(constants.CFG_PATH, f'{args.cfg}.yaml'))
    if args.device == 'gpu' and not is_available():
        raise ValueError('gpu is not available')
    train_data, val_data, test_data = load_datasets(args.dataset, config['input_shape'], config['latent_dim'],
                                                    config['age_dim'], args.sample_size, args.val_size, args.test_size,
                                                    args.splits_path, args.redo_splits, args.final, shuffle=True,
                                                    random_state=42)
    save_path = Path(constants.CHECKPOINT_PATH, args.dataset, args.cfg)
    run_name = f'e{args.epochs}'
    if args.run_name:
        run_name = f'{args.run_name}_{run_name}'
    save_path = save_path / run_name
    if not save_path.exists():
        save_path.mkdir(parents=True)
    train(config, train_data, val_data, args.batch_size, args.epochs, args.precision, args.log_interval,
          args.device, args.workers, args.no_sync, save_path)
