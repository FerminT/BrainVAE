from itertools import product
from lightning import seed_everything, Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.loggers.logger import DummyLogger
from pandas import DataFrame
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, mean_absolute_error
from sklearn.model_selection import KFold
from torch import device as dev, cuda
from tqdm import tqdm
from models.embedding_classifier import EmbeddingClassifier
from scripts.data_handler import get_loader
from scripts.constants import PARAM_GRID
from scripts.embedding_dataset import EmbeddingDataset
from scripts.utils import get_model_prediction
import wandb


def grid_search_cv(train_df, cfg_name, latent_dim, target_label, transform_fn, binary_classification, output_dim,
                   bin_centers, use_age, device, k_folds=10):
    print(f"Starting grid search with {k_folds}-fold cross validation...")
    print(f"Total configurations to test: {len(list(product(*PARAM_GRID.values())))}")
    best_auc, best_params = 0, None
    results = []
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    for params in tqdm(list(product(*PARAM_GRID.values())), desc="Grid Search"):
        lr, n_layers, batch_size, epochs = params
        param_dict = {
            'learning_rate': lr,
            'n_layers': n_layers,
            'batch_size': batch_size,
            'epochs': epochs
        }
        cfg_predictions = []
        cfg_labels = []
        for train_idx, val_idx in kf.split(train_df):
            train_fold = train_df.iloc[train_idx]
            val_fold = train_df.iloc[val_idx]
            train_dataset = EmbeddingDataset(train_fold, target=target_label, transform_fn=transform_fn)
            val_dataset = EmbeddingDataset(val_fold, target=target_label, transform_fn=transform_fn)
            classifier = train_classifier(train_dataset, val_dataset, cfg_name, latent_dim, output_dim,
                                          n_layers, bin_centers, use_age, batch_size, epochs, lr, device,
                                          log=False, seed=42)
            fold_preds, fold_lbls = test_classifier(classifier, val_dataset, binary_classification,
                                                    bin_centers, use_age, device)
            cfg_predictions.extend(fold_preds)
            cfg_labels.extend(fold_lbls)

        if binary_classification:
            auc = roc_auc_score(cfg_labels, cfg_predictions)
        else:
            corr, _ = pearsonr(cfg_predictions, cfg_labels)
            auc = abs(corr)
        results.append({
            'params': param_dict,
            'auc': auc
        })
        if auc > best_auc:
            best_auc = auc
            best_params = param_dict
        print(f"Config {param_dict}:\n AUC = {auc:.4f}\n----------------------------")
    return best_params, best_auc, results


def test_classifier(model, test_dataset, binary_classification, bin_centers, use_age, device):
    device = dev('cuda' if device == 'gpu' and cuda.is_available() else 'cpu')
    model.eval().to(device)
    predictions, labels = [], []
    for idx in tqdm(range(len(test_dataset)), desc='Evaluation'):
        z, target, age = test_dataset[idx]
        z = z.unsqueeze(dim=0).to(device)
        prediction = get_model_prediction(z, model, age, use_age, device, binary_classification, bin_centers)
        predictions.append(prediction)
        label = target.item() if binary_classification else (target.float().cpu() @ bin_centers).item()
        labels.append(label)
    return predictions, labels


def train_classifier(train_data, config_name, latent_dim, output_dim, n_layers, bin_centers, use_age,
                     batch_size, epochs, learning_rate, device, log, seed):
    seed_everything(seed, workers=True)
    wandb_logger = WandbLogger(name=f'classifier_{config_name}', project='BrainVAE', offline=True) \
        if log else DummyLogger()
    classifier = EmbeddingClassifier(input_dim=latent_dim, output_dim=output_dim, n_layers=n_layers,
                                     bin_centers=bin_centers, use_age=use_age, lr=learning_rate)
    train_dataloader = get_loader(train_data, batch_size=batch_size, shuffle=True)
    trainer = Trainer(max_epochs=epochs,
                      accelerator=device,
                      precision='bf16-mixed',
                      logger=wandb_logger,
                      )
    trainer.fit(classifier, train_dataloader)
    wandb.finish()
    return classifier


def compute_metrics(predictions, labels, binary_classification, results_dict):
    if binary_classification:
        predicted_classes = [1 if pred > 0.5 else 0 for pred in predictions]
        acc = accuracy_score(labels, predicted_classes)
        precision, recall = precision_score(labels, predicted_classes), recall_score(labels, predicted_classes)
        results_dict['Accuracy'].append(acc)
        results_dict['Precision'].append(precision)
        results_dict['Recall'].append(recall)
    else:
        mae = mean_absolute_error(labels, predictions)
        corr, p_value = pearsonr(predictions, labels)
        results_dict['MAE'].append(mae)
        results_dict['Corr'].append(corr)
        results_dict['p_value'].append(p_value)
    results_dict['Predictions'].append(predictions)
    results_dict['Labels'].append(labels)


def report_results(results_dict, target_label, name):
    model_predictions = results_dict.pop('Predictions')
    labels = results_dict.pop('Labels')
    results_df = DataFrame(results_dict)
    mean_df = results_df.mean(axis=0).to_frame(name='Mean')
    mean_df['SE'] = results_df.sem(axis=0)
    print(f'Predictions for {target_label} using {name} model')
    print(mean_df)
    return model_predictions, labels


def add_baseline_results(labels, binary_classification, baseline_results, rnd_gen):
    random_labels = labels.copy()
    if binary_classification:
        positive_proportion = sum(labels) / len(labels)
        random_labels = rnd_gen.binomial(1, positive_proportion, size=len(labels))
    else:
        rnd_gen.shuffle(random_labels)
    compute_metrics(random_labels, labels, binary_classification, baseline_results)
