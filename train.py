import argparse
from pathlib import Path
from models import icvae, loss
import yaml


def train(model, metadata, batch_size, lr, epochs, log_interval, loss_fn, device, save_path):
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
    parser.add_argument('--device', type=str, default='cpu', help='device (cuda or cpu)')
    parser.add_argument('--save_path', type=str, default='models', help='save path')

    args = parser.parse_args()
    metadata = Path('datasets', args.dataset, 'metadata', f'{args.dataset}_image_baseline_metadata.csv')
    save_path = Path(args.save_path, args.dataset)
    model_config = load_yaml(Path('cfg', args.cfg))
    model = icvae.ICVAE(**model_config)
    train(model, metadata, args.batch_size, args.lr, args.epochs, args.log_interval, loss.vae_loss,
          args.device, save_path)
