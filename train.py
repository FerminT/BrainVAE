import argparse
from pathlib import Path
from models import icvae, loss
from scripts import data
import yaml


def train(model, dataset, val_dataset, batch_size, lr, epochs, log_interval, loss_fn, device, save_path):
    pass


def load_yaml(filepath):
    with filepath.open('r') as file:
        return yaml.safe_load(file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ukbb', help='dataset name')
    parser.add_argument('--model', type=str, default='icvae', help='model name')
    parser.add_argument('--cfg', type=str, default='default.yaml', help='config file')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--log_interval', type=int, default=10, help='log interval')
    parser.add_argument('--sample_size', type=int, default=-1, help='number of samples to use')
    parser.add_argument('--val_size', type=float, default=0.2, help='validation size')
    parser.add_argument('--test_size', type=float, default=0.1, help='test size')
    parser.add_argument('--device', type=str, default='cpu', help='device (cuda or cpu)')
    parser.add_argument('--save_path', type=str, default='models', help='save path')

    args = parser.parse_args()
    datapath = Path('datasets', args.dataset)
    config = load_yaml(Path('cfg', args.cfg))
    train_data, val_data, test_data = data.load_datasets(datapath, config['input_shape'], args.sample_size,
                                                         args.val_size, args.test_size, shuffle=True, random_state=42)
    save_path = Path(args.save_path, args.dataset)
    model = icvae.ICVAE(**config)
    train(model, train_data, val_data, args.batch_size, args.lr, args.epochs, args.log_interval, loss.vae_loss,
          args.device, save_path)
