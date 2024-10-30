from pathlib import Path
from scripts.constants import CFG_PATH, CHECKPOINT_PATH, EVALUATION_PATH
from scripts.data_handler import (get_loader, get_datapath, load_set, create_test_splits, target_mapping,
                                  balance_dataset)
from scripts.embedding_dataset import EmbeddingDataset
from scripts.t1_dataset import T1Dataset
from scripts.utils import load_yaml, reconstruction_comparison_grid, init_embedding, subjects_embeddings, load_model
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer, seed_everything
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_absolute_error
from models.embedding_classifier import EmbeddingClassifier
from models.utils import get_latent_representation
from scipy.stats import pearsonr
from tqdm import tqdm
from numpy import array, random
from pandas import DataFrame, cut
from seaborn import scatterplot, kdeplot, set_theme, color_palette
from PIL import ImageDraw, ImageFont
import matplotlib.pyplot as plt
import torch
import wandb
import argparse


def save_predictions(df, predictions, labels, target_name, model_name):
    df['predicted'] = predictions
    df['label'] = labels
    df = df.drop(columns=['embedding'])
    df.to_csv(f'predictions_{model_name}_{target_name}.csv')


def predict_from_embeddings(embeddings_df, cfg, dataset, ukbb_size, val_size, latent_dim, age_range, bmi_range,
                            target_label, target_dataset, batch_size, epochs, n_iters, no_sync, device):
    train, val = create_test_splits(embeddings_df, dataset, val_size, ukbb_size, target_dataset, n_upsampled=180)
    transform_fn, output_dim, bin_centers = target_mapping(embeddings_df, target_label, age_range, bmi_range)
    binary_classification = output_dim == 1
    val_dataset = EmbeddingDataset(val, target=target_label, transform_fn=transform_fn)
    rnd_gen = random.default_rng(seed=42)
    random_seeds = [rnd_gen.integers(1, 1000) for _ in range(n_iters)]
    all_metrics, all_preds, labels = [], [], []
    for seed in random_seeds:
        train = train.sample(frac=1, replace=True, random_state=seed)
        train_dataset = EmbeddingDataset(train, target=target_label, transform_fn=transform_fn)
        classifier = train_classifier(train_dataset, val_dataset, cfg, latent_dim, output_dim, bin_centers,
                                      batch_size, epochs, device, no_sync, seed=42)
        results = test_classifier(classifier, val_dataset, binary_classification, bin_centers, device, seed=42)
        all_metrics.append(results[:3]), all_preds.append(results[3])
        labels = results[4]
    column_names = ['Accuracy', 'Precision', 'Recall'] if binary_classification else ['MAE', 'Corr', 'p_value']
    results_df = DataFrame(all_metrics, columns=column_names)
    mean_df = results_df.mean(axis=0).to_frame(name='Mean')
    mean_df['SE'] = results_df.sem(axis=0)
    print(mean_df)
    save_predictions(val, all_preds, labels, target_label, cfg)


def train_classifier(train_data, val_data, config_name, latent_dim, output_dim, bin_centers, batch_size, epochs,
                     device, no_sync, seed):
    seed_everything(seed, workers=True)
    wandb_logger = WandbLogger(name=f'classifier_{config_name}', project='BrainVAE', offline=no_sync)
    classifier = EmbeddingClassifier(input_dim=latent_dim, output_dim=output_dim, bin_centers=bin_centers)
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


def test_classifier(model, val_dataset, binary_classification, bin_centers, device, seed):
    seed_everything(seed, workers=True)
    device = torch.device('cuda' if device == 'gpu' and torch.cuda.is_available() else 'cpu')
    model.eval().to(device)
    predictions, labels = [], []
    for idx in tqdm(range(len(val_dataset))):
        z, target = val_dataset[idx]
        z = z.unsqueeze(dim=0).to(device)
        prediction = model(z)
        prediction = prediction.item() if binary_classification else (torch.exp(prediction.detach()) @ bin_centers).item()
        predictions.append(prediction)
        label = target.item() if binary_classification else (target @ bin_centers).item()
        labels.append(label)
    if binary_classification:
        predicted_classes = [1 if pred > 0.5 else 0 for pred in predictions]
        acc = accuracy_score(labels, predicted_classes)
        precision, recall = precision_score(labels, predicted_classes), recall_score(labels, predicted_classes)
        return acc, precision, recall, predictions, labels
    else:
        mae = mean_absolute_error(labels, predictions)
        corr, p_value = pearsonr(predictions, labels)
        return mae, corr, p_value, predictions, labels


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


def plot_embeddings(subjects_df, method, label, save_path, color_by=None, annotate_ids=False):
    if label not in subjects_df:
        raise ValueError(f'{label} not found in the dataframe')
    seed_everything(42, workers=True)
    save_path.mkdir(parents=True, exist_ok=True)
    components = init_embedding(method).fit_transform(array(subjects_df['embedding'].to_list()))
    subjects_df['emb_x'], subjects_df['emb_y'] = components[:, 0], components[:, 1]
    set_theme()
    fig, ax = plt.subplots(figsize=(10, 8))
    if color_by:
        color_by_is_float = subjects_df[color_by].dtype == 'float64'
        if color_by_is_float:
            subjects_df[color_by] = subjects_df[color_by].astype(int)
        scatter = scatterplot(data=subjects_df, x='emb_x', y='emb_y', hue=color_by, style=label, ax=ax, alpha=0.5,
                              size=.3, palette='viridis_r')
        handles_scatter, labels_scatter = scatter.get_legend_handles_labels()
        kdeplot(data=subjects_df, x='emb_x', y='emb_y', hue=label, fill=False, ax=ax, alpha=0.8)
        if color_by_is_float:
            norm = plt.Normalize(subjects_df[color_by].min(), subjects_df[color_by].max())
            sm = plt.cm.ScalarMappable(cmap='viridis_r', norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label(color_by)
        handles_kde = [plt.Line2D([0], [0], color=color_palette()[0]),
                       plt.Line2D([0], [0], color=color_palette()[1])]
        labels_kde = labels_scatter[-2:]
        ax.legend(handles_scatter[-2:] + handles_kde, labels_scatter[-2:] + labels_kde,
                  loc='lower center', ncol=2)
    else:
        if label == 'age_at_scan' or label == 'bmi':
            subjects_df[label] = cut(subjects_df[label], bins=3)
            subjects_df = subjects_df[subjects_df[label] != subjects_df[label].cat.categories[1]]
        scatterplot(data=subjects_df, x='emb_x', y='emb_y', hue=label, ax=ax, alpha=0.5, size=.3)
        kdeplot(data=subjects_df, x='emb_x', y='emb_y', hue=label, fill=False, ax=ax)
    if annotate_ids:
        for i, subject_id in enumerate(subjects_df.index):
            ax.annotate(subject_id, (components[i, 0], components[i, 1]), alpha=0.6)
    ax.set_title(f'Latent representations {method.upper()} embeddings')
    ax.axes.xaxis.set_visible(False), ax.axes.yaxis.set_visible(False)
    filename = save_path / f'latents_{method}_{label}.png'
    plt.savefig(filename, dpi=150)
    plt.show()
    print(f'Figure saved at {filename}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('weights', type=str, help='checkpoint file')
    parser.add_argument('--ckpt_dataset', type=str, default='general',
                        help='dataset on which the checkpoint file was trained')
    parser.add_argument('--dataset', type=str, default='general', help='dataset on which to train the predictor')
    parser.add_argument('--splits_path', type=str, default='splits', help='path to the data splits')
    parser.add_argument('--target', type=str, default='general', help='target dataset for predicting features')
    parser.add_argument('--cfg', type=str, default='default', help='config file used for the trained model')
    parser.add_argument('--device', type=str, default='cpu', help='device used for training and evaluation')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size used for training the age classifier')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs used for training the age classifier')
    parser.add_argument('--n_iters', type=int, default=3,
                        help='number of iterations (with different seeds) to evaluate the age classifier')
    parser.add_argument('--sample', type=int, default=0, help='subject id from which to reconstruct MRI data')
    parser.add_argument('--age', type=float, default=0.0, help='age of the subject to resample to, if using ICVAE')
    parser.add_argument('--manifold', type=str, default=None,
                        help='Method to use for manifold learning (PCA, MDS, tSNE, Isomap)')
    parser.add_argument('--label', type=str, default='age_at_scan',
                        help='label used for prediction and plotting latent representations'),
    parser.add_argument('--color_label', type=str, default=None,
                        help='label used for coloring the embeddings when doing manifold learning')
    parser.add_argument('--set', type=str, default='val', help='set to evaluate (val or test)')
    parser.add_argument('--balance', action='store_true', help='balance the dataset by age and sex')
    parser.add_argument('--ukbb_size', type=float, default=0.15,
                        help='size of the validation split constructed from the ukbb set to evaluate')
    parser.add_argument('--val_size', type=float, default=0.3,
                        help='size of the validation split constructed from the set to evaluate')
    parser.add_argument('--random_state', type=int, default=42, help='random state for reproducibility')
    parser.add_argument('--sync', action='store_false', help='sync to wandb')
    args = parser.parse_args()

    config = load_yaml(Path(CFG_PATH, f'{args.cfg}.yaml'))
    weights_path = Path(CHECKPOINT_PATH, args.ckpt_dataset, args.cfg, args.weights)
    save_path = Path(EVALUATION_PATH, args.dataset, args.set, args.cfg) / weights_path.parent.name
    save_path.mkdir(parents=True, exist_ok=True)

    datapath = get_datapath(args.dataset)
    embeddings_df = subjects_embeddings(weights_path, args.dataset, config['input_shape'], config['latent_dim'],
                                        args.set, datapath, args.splits_path, args.random_state, save_path)
    embeddings_df = embeddings_df[~embeddings_df[args.label].isna()]
    if args.balance:
        embeddings_df = balance_dataset(embeddings_df, args.label)
    print(embeddings_df.groupby(args.label)['age_at_scan'].describe())
    print(embeddings_df.groupby(args.label)['gender'].describe())

    data, age_range, bmi_range = load_set(args.dataset, args.set, args.splits_path, args.random_state)
    if args.sample == 0 and not args.manifold:
        predict_from_embeddings(embeddings_df, args.cfg, args.dataset, args.ukbb_size, args.val_size,
                                config['latent_dim'], age_range, bmi_range, args.label, args.target, args.batch_size,
                                args.epochs, args.n_iters, args.sync, args.device)
    else:
        if args.sample > 0:
            data, age_range, bmi_range = load_set(args.dataset, args.set, args.splits_path, args.random_state)
            if args.age > 0 and not age_range[0] < args.age < age_range[1]:
                print(f'age {args.age} is not within the training range of {age_range[0]} and {age_range[1]}')
            dataset = T1Dataset(config['input_shape'], datapath, data, config['latent_dim'], config['age_dim'],
                                age_range, bmi_range, testing=True)
            device = torch.device('cuda' if args.device == 'gpu' and torch.cuda.is_available() else 'cpu')
            model = load_model(weights_path, device)
            sample(model, dataset, args.age, args.sample, device, save_path)
        elif args.manifold:
            plot_embeddings(embeddings_df, args.manifold.lower(), args.label, save_path, color_by=args.color_label)
