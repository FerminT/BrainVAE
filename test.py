from pathlib import Path
from scripts.constants import DATA_PATH, CFG_PATH, CHECKPOINT_PATH, EVALUATION_PATH
from scripts.data_handler import load_metadata, T1Dataset, EmbeddingDataset, get_loader, age_to_tensor
from scripts.utils import load_yaml, reconstruction_comparison_grid, load_set, init_embedding, subjects_embeddings
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch import Trainer, seed_everything
from sklearn.model_selection import train_test_split
from models.age_classifier import AgeClassifier
from models.icvae import ICVAE
from models.utils import get_latent_representation
from scipy.stats import pearsonr
from tqdm import tqdm
from numpy import array
from pandas import cut
from seaborn import scatterplot, kdeplot
import matplotlib.pyplot as plt
import torch
import wandb
import argparse


def predict_from_embeddings(embeddings_df, cfg, val_size, latent_dim, batch_size, epochs, workers,
                            no_sync, device, save_path):
    save_path = save_path / 'age_classifier'
    save_path.mkdir(parents=True, exist_ok=True)
    train, val = train_test_split(embeddings_df, test_size=val_size, random_state=42)
    train_dataset = EmbeddingDataset(train, target='age_at_scan', transform_fn=age_to_tensor)
    val_dataset = EmbeddingDataset(val, target='age_at_scan', transform_fn=age_to_tensor)
    checkpoints = sorted(save_path.glob('*.ckpt'))
    if not checkpoints:
        train_classifier(train_dataset, val_dataset, cfg, latent_dim, batch_size, epochs, device, workers,
                         no_sync, save_path)
    else:
        print(f'Age classifier already trained, using {checkpoints[-1]}')
        model = AgeClassifier.load_from_checkpoint(checkpoints[-1])
        test_classifier(model, val_dataset, device)


def train_classifier(train_data, val_data, config_name, latent_dim, batch_size, epochs, device, workers,
                     no_sync, save_path):
    seed_everything(42, workers=True)
    wandb_logger = WandbLogger(name=f'ageclassifier_{config_name}', project='BrainVAE', offline=no_sync)
    checkpoint = ModelCheckpoint(dirpath=save_path, filename='{epoch:03d}-{val_mae:.2f}', monitor='val_mae',
                                 mode='min', save_top_k=2, save_last=True)
    early_stopping = EarlyStopping(monitor='val_mae', patience=5, mode='min')
    age_classifier = AgeClassifier(input_dim=latent_dim)
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
    predictions, labels = [], []
    for idx in tqdm(range(len(val_dataset))):
        z, target = val_dataset[idx]
        z = z.unsqueeze(dim=0).to(device)
        prediction = model(z).item()
        predictions.append(prediction)
        labels.append(target.item())
    corr, p_value = pearsonr(predictions, labels)
    print(f'Correlation between predictions and ages: {corr} (p-value: {p_value:.5f})')


def sample(model, dataset, age, subject_id, device, save_path):
    seed_everything(42, workers=True)
    save_path = save_path / 'samples'
    save_path.mkdir(parents=True, exist_ok=True)
    sample = dataset.get_subject(subject_id)
    device = torch.device('cuda' if device == 'gpu' and torch.cuda.is_available() else 'cpu')
    t1_img, _ = dataset.load_and_process_img(sample)
    t1_img = t1_img.unsqueeze(dim=0).to(device)
    z = get_latent_representation(t1_img, model.encoder)
    if age > 0.0:
        sample['age_at_scan'] = age
    age = dataset.age_mapping(sample['age_at_scan']).unsqueeze(dim=0)
    reconstructed = model.decoder(z, age.to(device))
    axes_comparisons, _ = reconstruction_comparison_grid(t1_img, reconstructed, 1, 80, 0)
    comparison = torch.cat(axes_comparisons, dim=2)
    comparison_img = wandb.Image(comparison).image
    comparison_img.save(save_path / f'{subject_id}_age_{int(sample["age_at_scan"])}.png')
    plt.imshow(comparison_img)
    plt.axis('off'), plt.xticks([]), plt.yticks([])
    plt.show()
    print(f'Reconstructed MRI saved at {save_path}')


def plot_embeddings(subjects_df, method, label, data_type, save_path, annotate_ids=False):
    if label not in subjects_df:
        raise ValueError(f'{label} not found in the dataframe')
    seed_everything(42, workers=True)
    save_path.mkdir(parents=True, exist_ok=True)
    if data_type == 'continuous':
        subjects_df[label] = cut(subjects_df[label], bins=3, labels=['low', 'middle', 'high'])
    components = init_embedding(method).fit_transform(array(subjects_df['embedding'].to_list()))
    subjects_df['emb_x'], subjects_df['emb_y'] = components[:, 0], components[:, 1]
    fig, ax = plt.subplots()
    # usar hexbin
    scatterplot(data=subjects_df, x='emb_x', y='emb_y', hue=label, ax=ax, alpha=0.3, size=.3)
    kdeplot(data=subjects_df, x='emb_x', y='emb_y', hue=label, fill=False, ax=ax)
    if annotate_ids:
        for i, subject_id in enumerate(subjects_df.index):
            ax.annotate(subject_id, (components[i, 0], components[i, 1]), alpha=0.6)
    ax.set_title(f'Latent representations {method.upper()} embeddings')
    ax.axes.xaxis.set_visible(False), ax.axes.yaxis.set_visible(False)
    filename = save_path / f'latents_{method}_{label}.png'
    plt.savefig(filename)
    plt.show()
    print(f'Figure saved at {filename}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('weights', type=str,
                        help='checkpoint file')
    parser.add_argument('--dataset', type=str, default='ukbb',
                        help='dataset name')
    parser.add_argument('--cfg', type=str, default='default',
                        help='config file used for the trained model')
    parser.add_argument('--device', type=str, default='gpu',
                        help='device used for training and evaluation')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch size used for training the age classifier')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs used for training the age classifier')
    parser.add_argument('--workers', type=int, default=12,
                        help='number of workers used for data loading when training the age classifier')
    parser.add_argument('--sample_size', type=int, default=-1,
                        help='number of samples used for training the model')
    parser.add_argument('--sample', type=int, default=0,
                        help='subject id from which to reconstruct MRI data')
    parser.add_argument('--age', type=float, default=0.0,
                        help='age of the subject to resample to, if using ICVAE')
    parser.add_argument('--manifold', type=str, default=None,
                        help='Method to use for manifold learning (PCA, MDS, tSNE, Isomap)')
    parser.add_argument('--label', type=str, default='age',
                        help='label used for plotting latent representations (age; gender; bmi)'),
    parser.add_argument('--data_type', type=str, default='continuous',
                        help='data type: either continuous or discrete')
    parser.add_argument('--set', type=str, default='val',
                        help='set to evaluate (val or test)')
    parser.add_argument('--val_size', type=float, default=0.1,
                        help='size of the validation set constructed from the set to evaluate')
    parser.add_argument('--no_sync', action='store_true',
                        help='do not sync to wandb')
    args = parser.parse_args()

    datapath = Path(DATA_PATH, args.dataset)
    config = load_yaml(Path(CFG_PATH, f'{args.cfg}.yaml'))
    _, age_range = load_metadata(datapath)
    if args.age > 0 and not age_range[0] < args.age < age_range[1]:
        print(f'age {args.age} is not within the training range of {age_range[0]} and {age_range[1]}')

    weights_path = Path(CHECKPOINT_PATH, args.dataset, args.cfg, args.weights)
    weights = next(weights_path.parent.glob(f'{weights_path.name}*'))
    save_path = Path(EVALUATION_PATH, args.dataset, args.set, args.cfg) / weights_path.parent.name
    data = load_set(datapath, args.sample_size, args.set)
    dataset = T1Dataset(config['input_shape'], datapath, data, conditional_dim=0, age_range=age_range,
                        one_hot_age=False, testing=True)

    save_path.mkdir(parents=True, exist_ok=True)
    model = ICVAE.load_from_checkpoint(weights)
    model.eval()
    device = torch.device('cuda' if args.device == 'gpu' and torch.cuda.is_available() else 'cpu')
    embeddings_df = subjects_embeddings(dataset, model, device, save_path)
    if args.sample == 0 and not args.manifold:
        predict_from_embeddings(embeddings_df, args.cfg, args.val_size, config['latent_dim'], args.batch_size,
                                args.epochs, args.workers, args.no_sync, args.device, save_path)
    else:
        if args.sample > 0:
            dataset = T1Dataset(config['input_shape'], datapath, data, config['conditional_dim'], age_range,
                                config['one_hot_age'], testing=True)
            sample(model, dataset, args.age, args.sample, args.device, save_path)
        elif args.manifold:
            plot_embeddings(embeddings_df, args.manifold.lower(), args.label, args.data_type, save_path)
