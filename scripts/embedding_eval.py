from lightning import seed_everything, Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.loggers.logger import DummyLogger
from pandas import DataFrame
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from torch import device as dev, cuda
from models.embedding_classifier import EmbeddingClassifier
from scripts.data_handler import get_loader
from scripts.embedding_dataset import EmbeddingDataset
from scripts.utils import get_model_prediction
import wandb


def test_on_val(train_df, val_size, cfg_name, latent_dim, target_label, transform_fn, output_dim,
                bin_centers, use_age, device, learning_rate, dropout, n_layers, batch_size, epochs):
    train, val = train_test_split(train_df, test_size=val_size, random_state=42)
    train_dataset = EmbeddingDataset(train, target=target_label, transform_fn=transform_fn)
    val_dataset = EmbeddingDataset(val, target=target_label, transform_fn=transform_fn)
    run_name = f'{target_label}_{cfg_name}_l{n_layers}_bs{batch_size}'
    if dropout > 0.0:
        run_name += f'_d{dropout}'
    classifier = train_classifier(train_dataset, val_dataset, run_name, latent_dim, output_dim,
                                  n_layers, bin_centers, use_age, batch_size, epochs, learning_rate, dropout, device,
                                  log=True, seed=42)
    return classifier


def test_classifier(model, test_dataset, binary_classification, bin_centers, use_age, device):
    device = dev('cuda' if device == 'gpu' and cuda.is_available() else 'cpu')
    model.eval().to(device)
    predictions, labels = [], []
    for idx in range(len(test_dataset)):
        z, target, age = test_dataset[idx]
        z = z.unsqueeze(dim=0).to(device)
        prediction = get_model_prediction(z, model, age, use_age, device, binary_classification, bin_centers)
        predictions.append(prediction)
        label = target.item() if binary_classification else (target.float().cpu() @ bin_centers).item()
        labels.append(label)
    return predictions, labels


def train_classifier(train_data, val_data, config_name, latent_dim, output_dim, n_layers, bin_centers, use_age,
                     batch_size, epochs, learning_rate, dropout, device, log, seed):
    seed_everything(seed, workers=True)
    wandb_logger = WandbLogger(name=f'classifier_{config_name}', project='BrainVAE', offline=False) \
        if log else DummyLogger()
    classifier = EmbeddingClassifier(input_dim=latent_dim, output_dim=output_dim, n_layers=n_layers,
                                     bin_centers=bin_centers, use_age=use_age, lr=learning_rate, dropout=dropout)
    train_dataloader = get_loader(train_data, batch_size=batch_size, shuffle=True)
    val_dataloader = get_loader(val_data, batch_size=batch_size, shuffle=False)
    trainer = Trainer(max_epochs=epochs,
                      accelerator=device,
                      precision='bf16-mixed',
                      logger=wandb_logger,
                      enable_checkpointing=False
                      )
    trainer.fit(classifier, train_dataloader, val_dataloader)
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
