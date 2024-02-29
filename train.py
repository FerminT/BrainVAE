from pathlib import Path
from scripts.data_handler import get_loader, load_datasets
from scripts.utils import load_yaml, load_architecture
from scripts.log import LogReconstructionsCallback
from scripts import constants
from torch.cuda import is_available
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch import Trainer, seed_everything
import wandb
import argparse


def train(config, train_data, val_data, batch_size, epochs, log_interval, device, no_sync, save_path):
    seed_everything(42, workers=True)
    model = load_architecture(config, len(train_data), epochs)
    train_loader = get_loader(train_data, batch_size, shuffle=False)
    val_loader = get_loader(val_data, batch_size, shuffle=False)
    wandb_logger = WandbLogger(name=f'{save_path.parent.name}_{save_path.name}', project='BrainVAE', offline=no_sync)
    checkpoint = ModelCheckpoint(dirpath=save_path, filename='{epoch:03d}-{val_loss:.2f}', monitor='val_loss',
                                          mode='min', save_top_k=10)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min')
    reconstruction = LogReconstructionsCallback(sample_size=8)
    trainer = Trainer(max_epochs=epochs,
                      accelerator=device,
                      precision='16-mixed',
                      logger=wandb_logger,
                      callbacks=[checkpoint, reconstruction, early_stopping, lr_monitor],
                      log_every_n_steps=min(log_interval, len(train_loader) // 10)
                      )
    checkpoints = sorted(save_path.glob('*.ckpt'))
    trainer.fit(model, train_loader, val_loader, ckpt_path=checkpoints[-1] if checkpoints else None)
    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ukbb', help='dataset name')
    parser.add_argument('--cfg', type=str, default='default.yaml', help='config file')
    parser.add_argument('--batch_size', type=int, default=6, help='batch size')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--log_interval', type=int, default=50, help='log interval')
    parser.add_argument('--sample_size', type=int, default=-1, help='number of samples to use')
    parser.add_argument('--val_size', type=float, default=0.1, help='validation size')
    parser.add_argument('--test_size', type=float, default=0.15, help='test size')
    parser.add_argument('--redo_splits', action='store_true', help='redo train/val/test splits')
    parser.add_argument('--no_sync', action='store_true', help='do not sync to wandb')
    parser.add_argument('--device', type=str, default='gpu', help='device (gpu or cpu)')
    parser.add_argument('--run_name', type=str, default='', help='(optional) add prefix to default run name')

    args = parser.parse_args()
    datapath = Path(constants.DATA_PATH, args.dataset)
    config = load_yaml(Path(constants.CFG_PATH, args.cfg))
    if args.device == 'gpu' and not is_available():
        raise ValueError('gpu is not available')
    train_data, val_data, test_data = load_datasets(datapath, config['input_shape'], config['conditional_dim'],
                                                    args.sample_size, args.val_size, args.test_size, args.redo_splits,
                                                    shuffle=True, random_state=42)
    save_path = Path(constants.CHECKPOINT_PATH, args.dataset, args.cfg.split('.')[0])
    run_name = f'e{args.epochs}'
    if args.sample_size != -1:
        run_name = f's{args.sample_size}_{run_name}'
    if args.run_name:
        run_name = f'{args.run_name}_{run_name}'
    save_path = save_path / run_name
    if not save_path.exists():
        save_path.mkdir(parents=True)
    train(config, train_data, val_data, args.batch_size, args.epochs, args.log_interval,
          args.device, args.no_sync, save_path)
