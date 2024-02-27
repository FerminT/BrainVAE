from pathlib import Path
from scripts.constants import DATA_PATH, CFG_PATH, CHECKPOINT_PATH, EVALUATION_PATH
from scripts.data_handler import load_metadata, T1Dataset, get_loader
from scripts.utils import load_yaml, get_splits_files
from torch.cuda import is_available
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch import Trainer, seed_everything
from models.age_classifier import AgeClassifier
import wandb
import argparse


def test(weights_path, config_name, dataset, latent_dim, batch_size, epochs, device, no_sync, save_path):
    seed_everything(42, workers=True)
    wandb_logger = WandbLogger(name=f'ageclassifier_{config_name}', project='BrainVAE', offline=no_sync)
    checkpoint = ModelCheckpoint(dirpath=save_path, filename='{epoch:03d}-{train_mae:.2f}', monitor='train_mae',
                                 mode='min', save_top_k=5)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    early_stopping = EarlyStopping(monitor='train_mae', patience=10, mode='min')
    age_classifier = AgeClassifier(weights_path, input_dim=latent_dim)
    dataloader = get_loader(dataset, batch_size=batch_size, shuffle=False)
    trainer = Trainer(max_epochs=epochs,
                      accelerator=device,
                      precision='16-mixed',
                      logger=wandb_logger,
                      callbacks=[checkpoint, early_stopping, lr_monitor]
                      )
    trainer.fit(age_classifier, dataloader)
    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('weights', type=str, help='checkpoint file')
    parser.add_argument('--dataset', type=str, default='ukbb', help='dataset name')
    parser.add_argument('--cfg', type=str, default='default', help='config file used for the trained model')
    parser.add_argument('--device', type=str, default='gpu', help='device used for training and evaluation')
    parser.add_argument('--batch_size', type=int, default=6, help='batch size used for training the age classifier')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs used for training the age classifier')
    parser.add_argument('--sample_size', type=int, default=-1, help='number of samples used for training the model')
    parser.add_argument('--set', type=str, default='val', help='set to evaluate (val or test)')
    parser.add_argument('--no_sync', action='store_true', help='do not sync to wandb')

    args = parser.parse_args()
    datapath = Path(DATA_PATH, args.dataset)
    config = load_yaml(Path(CFG_PATH, f'{args.cfg}.yaml'))
    _, age_range = load_metadata(datapath)
    _, val_csv, test_csv = get_splits_files(datapath, args.sample_size)
    if not val_csv.exists() or not test_csv.exists():
        raise ValueError(f'splits files for a sample size of {args.sample_size} do not exist')

    if args.device == 'gpu' and not is_available():
        raise ValueError('gpu is not available')

    if args.set == 'val':
        dataset = T1Dataset(config['input_shape'], datapath, val_csv, conditional_dim=1, age_range=age_range,
                            testing=True)
    else:
        dataset = T1Dataset(config['input_shape'], datapath, test_csv, conditional_dim=1, age_range=age_range,
                            testing=True)
    weights = Path(CHECKPOINT_PATH, args.dataset, args.cfg, args.weights)
    save_path = Path(EVALUATION_PATH, args.dataset, args.cfg)
    if not save_path.exists():
        save_path.mkdir(parents=True)
    test(weights, args.cfg, dataset, config['latent_dim'], args.batch_size, args.epochs,
         args.device, args.no_sync, save_path)
