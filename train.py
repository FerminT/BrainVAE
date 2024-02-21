from pathlib import Path
from scripts.data_handler import get_loader, load_datasets
from scripts.utils import load_yaml, load_architecture
from scripts.log import LogReconstructionsCallback
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, DeviceStatsMonitor
from lightning.pytorch import Trainer, seed_everything
import wandb
import argparse
import torch


def train(model_name, config, train_data, val_data, batch_size, epochs, device, no_sync, save_path):
    seed_everything(42, workers=True)
    model = load_architecture(model_name, config, len(train_data), epochs)
    train_loader = get_loader(train_data, batch_size, shuffle=False)
    val_loader = get_loader(val_data, batch_size, shuffle=False)
    wandb.init()
    wandb_logger = WandbLogger(name=f'{save_path.parent.name}_{save_path.name}', project='BrainVAE', offline=no_sync,
                               log_model='all')
    device_stats_callback = DeviceStatsMonitor()
    checkpoint_callback = ModelCheckpoint(dirpath=save_path, filename='best', monitor='val/loss', mode='min')
    reconstruction_callback = LogReconstructionsCallback(sample_size=8)
    trainer = Trainer(max_epochs=epochs,
                      accelerator=device,
                      logger=wandb_logger,
                      callbacks=[checkpoint_callback, reconstruction_callback, device_stats_callback],
                      deterministic=True)
    trainer.fit(model, train_loader, val_loader)
    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ukbb', help='dataset name')
    parser.add_argument('--model', type=str, default='icvae', help='model name')
    parser.add_argument('--cfg', type=str, default='default.yaml', help='config file')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--sample_size', type=int, default=-1, help='number of samples to use')
    parser.add_argument('--val_size', type=float, default=0.2, help='validation size')
    parser.add_argument('--test_size', type=float, default=0.1, help='test size')
    parser.add_argument('--redo_splits', action='store_true', help='redo train/val/test splits')
    parser.add_argument('--no_sync', action='store_true', help='do not sync to wandb')
    parser.add_argument('--device', type=str, default='cpu', help='device (gpu or cpu)')
    parser.add_argument('--save_path', type=str, default='checkpoints', help='save path')

    args = parser.parse_args()
    datapath = Path('datasets', args.dataset)
    config = load_yaml(Path('cfg', args.cfg))
    if args.device == 'gpu' and not torch.cuda.is_available():
        raise ValueError('gpu is not available')
    train_data, val_data, test_data = load_datasets(datapath, config['input_shape'], config['conditional_dim'],
                                                    args.sample_size, args.val_size, args.test_size, args.redo_splits,
                                                    args.device, shuffle=True, random_state=42)
    save_path = Path(args.save_path, args.dataset, args.cfg.split('.')[0])
    run_name = f'b{args.batch_size}_e{args.epochs}'
    if args.sample_size != -1:
        run_name = f's{args.sample_size}_{run_name}'
    save_path = save_path / run_name
    if not save_path.exists():
        save_path.mkdir(parents=True)
    train(args.model, config, train_data, val_data, args.batch_size, args.epochs, args.device, args.no_sync, save_path)
