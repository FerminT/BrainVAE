from pathlib import Path
from scripts.constants import DATA_PATH, CFG_PATH, CHECKPOINT_PATH, EVALUATION_PATH
from scripts.data_handler import get_loader, gender_to_onehot, load_set
from scripts.embedding_dataset import EmbeddingDataset
from scripts.t1_dataset import T1Dataset, age_to_tensor
from scripts.utils import load_yaml, reconstruction_comparison_grid, init_embedding, subjects_embeddings, load_model
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer, seed_everything
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_absolute_error
from models.embedding_classifier import EmbeddingClassifier
from models.utils import get_latent_representation
from scipy.stats import pearsonr
from tqdm import tqdm
from numpy import array, random
from pandas import concat, cut, DataFrame
from seaborn import scatterplot, kdeplot
from PIL import ImageDraw, ImageFont
import matplotlib.pyplot as plt
import torch
import wandb
import argparse


def predict_from_embeddings(embeddings_df, cfg, ukbb_size, val_size, latent_dim, label, target_dataset, data_type,
                            batch_size, epochs, n_iters, no_sync, device):
    embeddings_df = embeddings_df[~embeddings_df[label].isna()]
    train, val = train_test_split(embeddings_df[embeddings_df['dataset'] != 'ukbb'], test_size=val_size,
                                  random_state=42)
    train_ukbb, val_ukbb = train_test_split(embeddings_df[embeddings_df['dataset'] == 'ukbb'], test_size=ukbb_size,
                                            random_state=42)
    train = train.groupby('dataset')[['dataset']].apply(lambda x: x.sample(180, replace=True, random_state=42))
    train = concat([train, train_ukbb]).sample(frac=1, random_state=42)
    val = concat([val, val_ukbb]).sample(frac=1, random_state=42)
    if target_dataset != 'all':
        val = val[val['dataset'] == target_dataset]
    transform_fn = age_to_tensor if data_type == 'continuous' else gender_to_onehot
    train_dataset = EmbeddingDataset(train, target=label, transform_fn=transform_fn)
    val_dataset = EmbeddingDataset(val, target=label, transform_fn=transform_fn)
    rnd_gen = random.default_rng(42)
    random_seeds = [rnd_gen.integers(1, 100) for _ in range(n_iters)]
    all_results = []
    for seed in random_seeds:
        classifier = train_classifier(train_dataset, val_dataset, cfg, latent_dim, data_type, batch_size, epochs,
                                      device, no_sync, seed)
        results = test_classifier(classifier, val_dataset, data_type, device, seed)
        all_results.append(results)
    column_names = ['Accuracy', 'Precision', 'Recall'] if data_type == 'categorical' else ['MAE', 'Corr', 'p_value']
    results_df = DataFrame(all_results, columns=column_names)
    mean_df = results_df.mean(axis=0).to_frame(name='Mean')
    mean_df['Std'] = results_df.std(axis=0)
    print(mean_df)


def train_classifier(train_data, val_data, config_name, latent_dim, data_type, batch_size, epochs, device,
                     no_sync, seed):
    seed_everything(seed, workers=True)
    wandb_logger = WandbLogger(name=f'classifier_{config_name}', project='BrainVAE', offline=no_sync)
    classifier = EmbeddingClassifier(input_dim=latent_dim, data_type=data_type)
    train_dataloader = get_loader(train_data, batch_size=batch_size, shuffle=True)
    val_dataloader = get_loader(val_data, batch_size=batch_size, shuffle=False)
    trainer = Trainer(max_epochs=epochs,
                      accelerator=device,
                      precision='32',
                      logger=wandb_logger,
                      )
    trainer.fit(classifier, train_dataloader, val_dataloader)
    wandb.finish()
    return classifier


def test_classifier(model, val_dataset, data_type, device, seed):
    seed_everything(seed, workers=True)
    device = torch.device('cuda' if device == 'gpu' and torch.cuda.is_available() else 'cpu')
    model.eval().to(device)
    predictions, labels = [], []
    for idx in tqdm(range(len(val_dataset))):
        z, target = val_dataset[idx]
        z = z.unsqueeze(dim=0).to(device)
        prediction = model(z).item()
        prediction = (prediction > 0.5) if data_type == 'categorical' else prediction
        predictions.append(prediction)
        labels.append(target.item())
    if data_type == 'categorical':
        acc = accuracy_score(labels, predictions)
        precision, recall = precision_score(labels, predictions), recall_score(labels, predictions)
        return acc, precision, recall
    else:
        mae = mean_absolute_error(labels, predictions)
        corr, p_value = pearsonr(predictions, labels)
        return mae, corr, p_value


def sample(model, dataset, age, subject_id, device, save_path):
    seed_everything(42, workers=True)
    save_path = save_path / 'samples'
    save_path.mkdir(parents=True, exist_ok=True)
    sample = dataset.get_subject(subject_id)
    t1_img, _ = dataset.load_and_process_img(sample)
    t1_img = t1_img.unsqueeze(dim=0).to(device)
    z = get_latent_representation(t1_img, model.encoder)
    if age > 0.0:
        sample['age_at_scan'] = age
    age = dataset.age_mapping(sample['age_at_scan']).unsqueeze(dim=0)
    reconstructed = model.decoder(z, age.to(device))
    axes_comparisons, _ = reconstruction_comparison_grid(t1_img, reconstructed, 1, 50, 0)
    comparison = torch.cat(axes_comparisons, dim=2)
    comparison_img = wandb.Image(comparison).image
    draw, font = ImageDraw.Draw(comparison_img), ImageFont.truetype("LiberationSans-Regular.ttf", 25)
    draw.text((225, 183), f'Age: {int(sample["age_at_scan"])}', (255, 255, 255), font=font)
    comparison_img_name = f'{subject_id}_age_{int(sample["age_at_scan"])}.png'
    comparison_img.save(save_path / comparison_img_name)
    plt.imshow(comparison_img)
    plt.axis('off'), plt.xticks([]), plt.yticks([])
    plt.show()
    print(f'Reconstructed MRI saved at {save_path / comparison_img_name}')


def plot_embeddings(subjects_df, method, label, data_type, save_path, annotate_ids=False):
    if label not in subjects_df:
        raise ValueError(f'{label} not found in the dataframe')
    seed_everything(42, workers=True)
    save_path.mkdir(parents=True, exist_ok=True)
    if data_type == 'continuous':
        subjects_df[label] = cut(subjects_df[label], bins=3)
        # remove middle category
        subjects_df = subjects_df[subjects_df[label] != subjects_df[label].cat.categories[1]]
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
    parser.add_argument('--dataset', type=str, default='all',
                        help='dataset name')
    parser.add_argument('--target', type=str, default='all',
                        help='target dataset for predicting features')
    parser.add_argument('--cfg', type=str, default='default',
                        help='config file used for the trained model')
    parser.add_argument('--device', type=str, default='cpu',
                        help='device used for training and evaluation')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch size used for training the age classifier')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs used for training the age classifier')
    parser.add_argument('--n_iters', type=int, default=10,
                        help='number of iterations (with different seeds) to evaluate the age classifier')
    parser.add_argument('--sample', type=int, default=0,
                        help='subject id from which to reconstruct MRI data')
    parser.add_argument('--age', type=float, default=0.0,
                        help='age of the subject to resample to, if using ICVAE')
    parser.add_argument('--manifold', type=str, default=None,
                        help='Method to use for manifold learning (PCA, MDS, tSNE, Isomap)')
    parser.add_argument('--label', type=str, default='age_at_scan',
                        help='label used for prediction and plotting latent representations (age; gender; bmi)'),
    parser.add_argument('--data_type', type=str, default='continuous',
                        help='data type: either continuous or categorical')
    parser.add_argument('--set', type=str, default='val',
                        help='set to evaluate (val or test)')
    parser.add_argument('--ukbb_size', type=float, default=0.15,
                        help='size of the validation split constructed from the ukbb set to evaluate')
    parser.add_argument('--val_size', type=float, default=0.3,
                        help='size of the validation split constructed from the set to evaluate')
    parser.add_argument('--random_state', type=int, default=42,
                        help='random state for reproducibility')
    parser.add_argument('--sync', action='store_false',
                        help='sync to wandb')
    args = parser.parse_args()

    config = load_yaml(Path(CFG_PATH, f'{args.cfg}.yaml'))
    weights_path = Path(CHECKPOINT_PATH, args.dataset, args.cfg, args.weights)
    save_path = Path(EVALUATION_PATH, args.dataset, args.set, args.cfg) / weights_path.parent.name
    save_path.mkdir(parents=True, exist_ok=True)

    datapath = Path(DATA_PATH)
    embeddings_df = subjects_embeddings(weights_path, config['input_shape'], config['latent_dim'], args.set,
                                        datapath, args.random_state, save_path)
    if args.sample == 0 and not args.manifold:
        predict_from_embeddings(embeddings_df, args.cfg, args.ukbb_size, args.val_size, config['latent_dim'],
                                args.label, args.target, args.data_type, args.batch_size, args.epochs, args.n_iters,
                                args.sync, args.device)
    else:
        if args.sample > 0:
            data, age_range = load_set(args.dataset, args.split, args.random_state)
            if args.age > 0 and not age_range[0] < args.age < age_range[1]:
                print(f'age {args.age} is not within the training range of {age_range[0]} and {age_range[1]}')
            dataset = T1Dataset(config['input_shape'], datapath, data, config['latent_dim'], config['conditional_dim'],
                                age_range, config['invariant'], testing=True)
            device = torch.device('cuda' if args.device == 'gpu' and torch.cuda.is_available() else 'cpu')
            model = load_model(weights_path, device)
            sample(model, dataset, args.age, args.sample, device, save_path)
        elif args.manifold:
            plot_embeddings(embeddings_df, args.manifold.lower(), args.label, args.data_type, save_path)
