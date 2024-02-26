import argparse
from pathlib import Path
from scripts.constants import DATA_PATH, CFG_PATH, CHECKPOINT_PATH
from scripts.data_handler import load_metadata, T1Dataset
from scripts.utils import load_yaml, get_splits_files


def test(weights_path, val_data, test_data, device):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('weights', type=str, help='checkpoint file')
    parser.add_argument('--dataset', type=str, default='ukbb', help='dataset name')
    parser.add_argument('--cfg', type=str, default='default', help='config file used for the trained model')
    parser.add_argument('--device', type=str, default='cpu', help='device used for training and evaluation')
    parser.add_argument('--sample_size', type=int, default=-1, help='number of samples used for training the model')

    args = parser.parse_args()
    datapath = Path(DATA_PATH, args.dataset)
    config = load_yaml(Path(CFG_PATH, f'{args.cfg}.yaml'))
    _, age_range = load_metadata(datapath)
    _, val_csv, test_csv = get_splits_files(datapath, args.sample_size)
    if not val_csv.exists() or not test_csv.exists():
        raise ValueError(f'splits files for a sample size of {args.sample_size} do not exist')

    val_dataset = T1Dataset(config['input_shape'], datapath, val_csv, config['conditional_dim'], age_range,
                            testing=True)
    test_dataset = T1Dataset(config['input_shape'], datapath, test_csv, config['conditional_dim'], age_range,
                             testing=True)
    weights = Path(CHECKPOINT_PATH, args.dataset, args.cfg, args.weights)
    test(weights, val_dataset, test_dataset, args.device)
