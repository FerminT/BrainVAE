import pandas as pd
import yaml
from os import cpu_count
from numpy import inf, arange, zeros, abs
from pandas import concat
from functools import partial
from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.model_selection import train_test_split
from scripts.t1_dataset import T1Dataset, soft_label, label_to_onehot
from scripts import constants


def get_loader(dataset, batch_size, shuffle, num_workers=4):
    num_workers = min(cpu_count(), num_workers)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def load_metadata(datapath):
    metadata = pd.read_csv(datapath / constants.METADATA_PATH / f'{datapath.name}_image_baseline_metadata.csv')
    age_range = [int(metadata['age_at_scan'].min()), round(metadata['age_at_scan'].max() + 0.5)]
    bmi_range = [inf, -inf]
    if metadata['bmi'].notna().any():
        bmi_range = [int(metadata['bmi'].min()), round(metadata['bmi'].max() + 0.5)]
    return metadata, age_range, bmi_range


def load_datasets(dataset, input_shape, latent_dim, age_dim, invariance, sample_size, val_size, test_size, splits_path,
                  redo_splits, final_run, shuffle, random_state):
    datapath = get_datapath(dataset)
    datasets = get_datasets(dataset)
    train, val, test, age_range, bmi_range = combine_datasets(datasets, invariance, sample_size, val_size, test_size,
                                                              splits_path, redo_splits, final_run, shuffle,
                                                              random_state)
    train_dataset = T1Dataset(input_shape, datapath, train, latent_dim, age_dim, age_range, bmi_range, testing=False)
    val_dataset = T1Dataset(input_shape, datapath, val, latent_dim, age_dim, age_range, bmi_range, testing=True)
    test_dataset = T1Dataset(input_shape, datapath, test, latent_dim, age_dim, age_range, bmi_range, testing=True)
    return train_dataset, val_dataset, test_dataset


def combine_datasets(datasets, invariance, sample_size, val_size, test_size, splits_path, redo_splits, final_run,
                     shuffle, random_state):
    train_datasets, val_datasets, test_datasets = [], [], []
    age_range, bmi_range = [inf, -inf], [inf, -inf]
    for dataset in datasets:
        metadata, dataset_age_range, dataset_bmi_range = load_metadata(dataset)
        train, val, test = load_splits(dataset, invariance, metadata, sample_size, val_size, test_size, splits_path,
                                       redo_splits, final_run, shuffle=shuffle, random_state=random_state)
        age_range = [min(age_range[0], dataset_age_range[0]), max(age_range[1], dataset_age_range[1])]
        bmi_range = [min(bmi_range[0], dataset_bmi_range[0]), max(bmi_range[1], dataset_bmi_range[1])]
        train_datasets.append(train)
        val_datasets.append(val)
        test_datasets.append(test)
    train, val, test = pd.concat(train_datasets), pd.concat(val_datasets), pd.concat(test_datasets)
    train, val, test = (shuffle_data(train, random_state), shuffle_data(val, random_state),
                        shuffle_data(test, random_state))
    return train, val, test, age_range, bmi_range


def load_splits(datapath, invariance, metadata, sample_size, val_size, test_size, splits_path, redo, final_run, shuffle,
                random_state):
    train_csv, val_csv, test_csv = get_splits_files(datapath, splits_path)
    if train_csv.exists() and val_csv.exists() and test_csv.exists() and not redo:
        train = pd.read_csv(train_csv)
        val = pd.read_csv(val_csv)
        test = pd.read_csv(test_csv)
    else:
        train, val, test = generate_splits(metadata, val_size, test_size, shuffle, random_state)
        train_csv.absolute().parent.mkdir(parents=True, exist_ok=True)
        train.to_csv(train_csv, index=False)
        val.to_csv(val_csv, index=False)
        test.to_csv(test_csv, index=False)
    if invariance and invariance == 'bmi':
        train = train[train['bmi'].notna()]
        val = val[val['bmi'].notna()]
        test = test[test['bmi'].notna()]
    if final_run:
        train = pd.concat([train, val])
        val = test
    if not train.empty and datapath.name != 'ukbb' and sample_size:
        train = train.sample(sample_size, random_state=random_state, replace=True)
    return train, val, test


def generate_splits(data, val_size, test_size, shuffle, random_state):
    data = preprocess(data)
    train, val_test = train_test_split(data, test_size=val_size + test_size, shuffle=shuffle, random_state=random_state)
    val, test = train_test_split(val_test, test_size=test_size / (val_size + test_size), shuffle=shuffle,
                                 random_state=random_state)
    return train, val, test


def shuffle_data(data, random_state):
    return data.sample(frac=1, random_state=random_state)


def preprocess(data):
    data = data.drop_duplicates(subset='subject_id')
    return data


def get_splits_files(datapath, splits_path):
    splits_path = datapath / splits_path
    train_csv, val_csv, test_csv = splits_path / 'train.csv', splits_path / 'val.csv', splits_path / 'test.csv'
    return train_csv, val_csv, test_csv


def get_datasets(dataset):
    datapath = Path(constants.DATA_PATH)
    if not is_single_dataset(dataset):
        datasets = [d for d in (datapath / dataset).iterdir() if d.is_dir()]
    else:
        datasets = [datapath / dataset]
    return datasets


def get_datapath(dataset):
    datapath = Path(constants.DATA_PATH)
    if not is_single_dataset(dataset):
        datapath = datapath / dataset
    else:
        datapath = datapath / Path(dataset).parent
    return datapath


def is_single_dataset(dataset):
    return len(Path(dataset).parts) > 1


def load_set(dataset, split, invariance, splits_path, random_state):
    datasets = get_datasets(dataset)
    train, val, test, age_range, bmi_range = combine_datasets(datasets, invariance, sample_size=None,
                                                              val_size=None, test_size=None,
                                                              splits_path=splits_path, redo_splits=False,
                                                              final_run=False, shuffle=True, random_state=random_state)
    if split == 'val':
        data = val
    elif split == 'test':
        data = test
    else:
        data = train
    return data, age_range, bmi_range


def upsample_datasets(train, n_upsampled):
    for dataset in train['dataset'].unique():
        dataset_samples = train[train['dataset'] == dataset]
        n_samples = len(dataset_samples)
        if n_samples < n_upsampled:
            resampled = dataset_samples.sample(n=n_upsampled - n_samples, replace=True, random_state=42)
            train = concat([train, resampled])
    return train


def create_test_splits(embeddings_df, dataset, val_size, ukbb_size, target_dataset, n_upsampled):
    if dataset == 'general':
        train, val = train_test_split(embeddings_df[embeddings_df['dataset'] != 'ukbb'], test_size=val_size,
                                      random_state=42)
        train_ukbb, val_ukbb = train_test_split(embeddings_df[embeddings_df['dataset'] == 'ukbb'], test_size=ukbb_size,
                                                random_state=42)
        train = upsample_datasets(train, n_upsampled=n_upsampled)
        train = concat([train, train_ukbb]).sample(frac=1, random_state=42)
        val = concat([val, val_ukbb]).sample(frac=1, random_state=42)
    else:
        train, val = train_test_split(embeddings_df, test_size=val_size, random_state=42)
    if target_dataset != 'general' and target_dataset != 'diseased':
        val = val[val['dataset'] == target_dataset]
    return train, val


def balance_dataset(embeddings_df, group_label):
    minority_label = embeddings_df[group_label].value_counts().idxmin()
    majority_label = embeddings_df[group_label].value_counts().idxmax()
    embeddings_df_int_age = embeddings_df.copy()
    embeddings_df_int_age['age_at_scan'] = embeddings_df_int_age['age_at_scan'].astype(int)
    print('Balancing dataset...')
    balanced_indices = get_balanced_indices(embeddings_df_int_age, group_label, minority_label, majority_label,
                                            ['age_at_scan', 'gender'])
    embeddings_df = embeddings_df.loc[balanced_indices]
    return embeddings_df


def get_balanced_indices(embeddings_df, group_label, minority_label, majority_label, balance_by):
    embeddings_df['balance_by'] = embeddings_df[balance_by].astype(str).agg('-'.join, axis=1)
    minority = embeddings_df[embeddings_df[group_label] == minority_label]
    majority = embeddings_df[embeddings_df[group_label] == majority_label]
    minority_groups = list(minority['balance_by'].unique())
    minority_dist = minority['balance_by'].value_counts(sort=False).values
    current_balance = zeros(len(minority_groups))
    majority = majority[majority['balance_by'].isin(minority_groups)]
    matched = []
    while len(matched) < len(minority):
        subject_found = None
        imbalance_ratio = inf
        for _, subject in majority.iterrows():
            subject_group = minority_groups.index(subject['balance_by'])
            current_balance[subject_group] += 1
            if abs(current_balance - minority_dist).sum() < imbalance_ratio:
                subject_found = subject
                imbalance_ratio = abs(current_balance - minority_dist).sum()
            current_balance[subject_group] -= 1
        subject_group = minority_groups.index(subject_found['balance_by'])
        current_balance[subject_group] += 1
        matched.append(subject_found.name)
        majority = majority[majority.index != subject_found.name]
    balanced_majority = embeddings_df.loc[matched]
    balanced_indices = concat([minority, balanced_majority]).index
    return balanced_indices


def target_mapping(embeddings_df, label, age_range, bmi_range):
    if label == 'age_at_scan':
        transform_fn = partial(soft_label, lower=age_range[0], upper=age_range[1])
        output_dim = age_range[1] - age_range[0]
        data_range = age_range
    elif label == 'bmi':
        transform_fn = partial(soft_label, lower=bmi_range[0], upper=bmi_range[1])
        output_dim = bmi_range[1] - bmi_range[0]
        data_range = bmi_range
    else:
        labels = list(embeddings_df[label].unique())
        transform_fn = partial(label_to_onehot, labels=labels)
        output_dim = 1
        data_range = [0, 1]
    bin_centers = data_range[0] + 1.0 / 2 + 1.0 * arange(data_range[1] - data_range[0])
    return transform_fn, output_dim, bin_centers


def save_predictions(df, predictions, labels, target_name, params, save_path):
    preds_dict = {}
    for i, (pred, label) in enumerate(zip(predictions, labels)):
        preds_dict[f'pred_{i}'] = pred
        preds_dict[f'label_{i}'] = label
    predictions_df = pd.DataFrame(preds_dict, index=df.index)
    df = df.drop(columns=['embedding'])
    df = pd.concat([df, predictions_df], axis=1)
    if not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path / f'{target_name}_predictions.csv')
    with open(save_path / f'{target_name}_params.yaml', 'w') as file:
        yaml.dump(params, file)
