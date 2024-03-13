from pathlib import Path
from scripts.constants import DATA_PATH, CFG_PATH, CHECKPOINT_PATH, EVALUATION_PATH
from scripts.data_handler import load_metadata, T1Dataset, get_loader
from scripts.utils import load_yaml, get_splits_files, reconstruction_comparison_grid
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch import Trainer, seed_everything
from sklearn.model_selection import train_test_split
from models.age_classifier import AgeClassifier
from models.icvae import ICVAE
from models.utils import get_latent_representation
from scipy.stats import pearsonr
from pandas import read_csv
from numpy import array
from sklearn.decomposition import PCA
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import wandb
import argparse


def train_classifier(weights_path, config_name, train_data, val_data, latent_dim, batch_size, epochs, device, workers,
                     no_sync, save_path):
    seed_everything(42, workers=True)
    wandb_logger = WandbLogger(name=f'ageclassifier_{config_name}', project='BrainVAE', offline=no_sync)
    checkpoint = ModelCheckpoint(dirpath=save_path, filename='{epoch:03d}-{val_mae:.2f}', monitor='val_mae',
                                 mode='min', save_top_k=2)
    early_stopping = EarlyStopping(monitor='val_mae', patience=5, mode='min')
    age_classifier = AgeClassifier(weights_path, input_dim=latent_dim)
    train_dataloader = get_loader(train_data, batch_size=batch_size, shuffle=False, num_workers=workers)
    val_dataloader = get_loader(val_data, batch_size=batch_size, shuffle=False, num_workers=workers)
    trainer = Trainer(max_epochs=epochs,
                      accelerator=device,
                      precision='16-mixed',
                      logger=wandb_logger,
                      callbacks=[checkpoint, early_stopping]
                      )
    trainer.fit(age_classifier, train_dataloader, val_dataloader)
    wandb.finish()
    test_classifier(age_classifier, val_data, device)


def test_classifier(model, val_dataset, device):
    seed_everything(42, workers=True)
    device = torch.device('cuda' if device == 'gpu' and torch.cuda.is_available() else 'cpu')
    model.eval().to(device)
    predictions, ages = [], []
    for idx in tqdm(range(len(val_dataset))):
        x, _, age = val_dataset[idx]
        x = x.unsqueeze(dim=0).to(device)
        prediction = model(x).item()
        predictions.append(prediction)
        ages.append(age.item())
    corr, p_value = pearsonr(predictions, ages)
    print(f'correlation between predictions and ages: {corr} (p-value: {p_value:.5f})')


def sample(weights_path, dataset, age, subject_id, device, save_path):
    seed_everything(42, workers=True)
    sample = dataset.get_subject(subject_id)
    model = ICVAE.load_from_checkpoint(weights_path)
    model.eval()
    device = torch.device('cuda' if device == 'gpu' and torch.cuda.is_available() else 'cpu')
    t1_img, _ = dataset.load_and_process_img(sample)
    t1_img = t1_img.unsqueeze(dim=0).to(device)
    z = get_latent_representation(t1_img, model.encoder)
    if age > 0.0:
        sample['age_at_scan'] = age
    age = dataset.age_mapping(sample['age_at_scan']).unsqueeze(dim=0)
    reconstructed = model.decoder(z, age.to(device))
    comparison_grids = reconstruction_comparison_grid(t1_img, reconstructed, 1, 80, 0)
    for i, img in enumerate(comparison_grids[0]):
        wandb.Image(img).image.save(save_path / f'{subject_id}_age_{int(sample["age_at_scan"])}_axis_{i}.png')
    print(f'reconstructed MRI saved at {save_path}')


def pca_latent_dimension(weights_path, dataset, device, save_path):
    seed_everything(42, workers=True)
    model = ICVAE.load_from_checkpoint(weights_path)
    model.eval()
    device = torch.device('cuda' if device == 'gpu' and torch.cuda.is_available() else 'cpu')
    pca = PCA(n_components=2)
    latent_representations = []
    for idx in tqdm(range(len(dataset))):
        t1_img, _, _ = dataset[idx]
        t1_img = t1_img.unsqueeze(dim=0).to(device)
        z = get_latent_representation(t1_img, model.encoder)
        latent_representations.append(z.cpu().detach().numpy())
    latent_representations = array(latent_representations).reshape(len(latent_representations), -1)
    pca.fit(latent_representations)
    transformed = pca.transform(latent_representations)
    fig, ax = plt.subplots()
    ax.scatter(transformed[:, 0], transformed[:, 1])
    ax.set_title('PCA of latent representations')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    plt.show()
    plt.savefig(save_path / 'pca_latent_representations.png')
    print(f'PCA of latent representations saved at {save_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('weights', type=str, help='checkpoint file')
    parser.add_argument('--dataset', type=str, default='ukbb', help='dataset name')
    parser.add_argument('--cfg', type=str, default='default',
                        help='config file used for the trained model')
    parser.add_argument('--device', type=str, default='gpu',
                        help='device used for training and evaluation')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch size used for training the age classifier')
    parser.add_argument('--epochs', type=int, default=15,
                        help='number of epochs used for training the age classifier')
    parser.add_argument('--workers', type=int, default=12,
                        help='number of workers used for data loading when training the age classifier')
    parser.add_argument('--sample_size', type=int, default=-1,
                        help='number of samples used for training the model')
    parser.add_argument('--sample', type=int, default=0,
                        help='subject id from which to reconstruct MRI data')
    parser.add_argument('--age', type=float, default=0.0,
                        help='age of the subject to resample to, if using ICVAE')
    parser.add_argument('--pca', action='store_true', help='perform PCA on the latent representations')
    parser.add_argument('--set', type=str, default='val',
                        help='set to evaluate (val or test)')
    parser.add_argument('--val_size', type=float, default=0.1,
                        help='size of the validation set constructed from the set to evaluate')
    parser.add_argument('--no_sync', action='store_true', help='do not sync to wandb')

    args = parser.parse_args()
    datapath = Path(DATA_PATH, args.dataset)
    config = load_yaml(Path(CFG_PATH, f'{args.cfg}.yaml'))
    _, age_range = load_metadata(datapath)
    if args.age > 0 and not age_range[0] < args.age < age_range[1]:
        print(f'age {args.age} is not within the training range of {age_range[0]} and {age_range[1]}')
    train_csv, val_csv, test_csv = get_splits_files(datapath, args.sample_size)
    if not (train_csv.exists() and val_csv.exists() and test_csv.exists()):
        raise ValueError(f'splits files for a sample size of {args.sample_size} do not exist')

    weights = Path(CHECKPOINT_PATH, args.dataset, args.cfg, args.weights)
    save_path = Path(EVALUATION_PATH, args.dataset, args.cfg) / weights.parent.name
    if not save_path.exists():
        save_path.mkdir(parents=True)

    if args.set == 'val':
        data = read_csv(val_csv)
    elif args.set == 'test':
        data = read_csv(test_csv)
    else:
        data = read_csv(train_csv)
    if args.sample > 0:
        save_path = save_path / 'samples'
        save_path.mkdir(exist_ok=True)
        dataset = T1Dataset(config['input_shape'], datapath, data, config['conditional_dim'], age_range,
                            config['one_hot_age'], testing=True)
        sample(weights, dataset, args.age, args.sample, args.device, save_path)
    elif args.pca:
        dataset = T1Dataset(config['input_shape'], datapath, data, config['conditional_dim'], age_range,
                            config['one_hot_age'], testing=True)
        pca_latent_dimension(weights, dataset, args.device, save_path)
    else:
        save_path = save_path / 'age_classifier'
        save_path.mkdir(exist_ok=True)
        train, val = train_test_split(data, test_size=args.val_size, random_state=42)
        train_dataset = T1Dataset(config['input_shape'], datapath, train, 0, age_range, one_hot_age=False,
                                  testing=True)
        val_dataset = T1Dataset(config['input_shape'], datapath, val, 0, age_range, one_hot_age=False,
                                testing=True)
        checkpoints = sorted(save_path.glob('*.ckpt'))
        if not checkpoints:
            train_classifier(weights, args.cfg, train_dataset, val_dataset, config['latent_dim'], args.batch_size,
                             args.epochs, args.device, args.workers, args.no_sync, save_path)
        else:
            print(f'age classifier already trained, using {checkpoints[-1]}')
            model = AgeClassifier.load_from_checkpoint(checkpoints[-1])
            test_classifier(model, val_dataset, args.device)
