import argparse
from pathlib import Path
from model import icvae, loss


def train(model, metadata, batch_size, lr, epochs, log_interval, loss_fn, device, save_path):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ukbb', help='dataset name')
    parser.add_argument('--input_shape', type=tuple, default=(160, 192, 160), help='input shape')
    parser.add_argument('--latent_dim', type=int, default=354, help='latent dimension')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--log_interval', type=int, default=10, help='log interval')
    parser.add_argument('--kernel_size', type=int, default=3, help='kernel size')
    parser.add_argument('--padding', type=int, default=1, help='padding')
    parser.add_argument('--stride', type=int, default=1, help='stride')
    parser.add_argument('--pooling_kernel', type=int, default=2, help='pooling kernel')
    parser.add_argument('--pooling_stride', type=int, default=2, help='pooling stride')
    parser.add_argument('--last_kernel', type=int, default=1, help='last kernel size')
    parser.add_argument('--last_padding', type=int, default=0, help='last padding')
    parser.add_argument('--channels', type=tuple, default=(32, 64, 128, 256, 256, 64), help='channels')
    parser.add_argument('--device', type=str, default='cpu', help='device (cuda or cpu)')
    parser.add_argument('--save_path', type=str, default='model', help='save path')

    args = parser.parse_args()
    metadata = Path('datasets', args.dataset, 'metadata', f'{args.dataset}_image_baseline_metadata.csv')
    save_path = Path(args.save_path, args.dataset)
    model = icvae.ICVAE(args.input_shape, args.latent_dim, args.kernel_size, args.padding, args.stride,
                        args.pooling_kernel, args.pooling_stride, args.last_kernel, args.last_padding, args.channels)
    train(model, metadata, args.batch_size, args.lr, args.epochs, args.log_interval, loss.vae_loss,
          args.device, save_path)
