from pathlib import Path
from models import icvae, losses
from scripts import log
from scripts.data_handler import get_loader, load_datasets
from torchvision.utils import save_image
from scripts.utils import load_yaml
import argparse
import torch


def train(model_name, config, train_data, val_data, batch_size, lr, epochs, log_interval, device, run_name, save_path):
    model = getattr(icvae, model_name)(**config['params'])
    model.to(device)
    optimizer = getattr(torch.optim, config['optimizer'].upper())(model.parameters(), lr=lr)
    criterion = getattr(losses, config['loss'])
    train_loader = get_loader(train_data, batch_size, shuffle=False)
    val_loader = get_loader(val_data, batch_size, shuffle=False)
    log.init('BrainVAE', run_name, config['params']['latent_dim'], lr, batch_size, epochs, len(train_data))
    for epoch in range(epochs):
        model.train()
        train_rcon_loss, train_prior_loss = 0, 0
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            recon_loss, prior_loss = criterion(recon_batch, data, mu, logvar)
            loss = recon_loss + prior_loss
            loss.backward()
            train_rcon_loss += recon_loss.item()
            train_prior_loss += prior_loss.item()
            optimizer.step()
            if batch_idx % log_interval == 0:
                log.step({'train': {'batch': batch_idx, 'reconstruction_loss': recon_loss, 'prior_loss': prior_loss}})
        train_rcon_loss /= len(train_loader.dataset)
        train_prior_loss /= len(train_loader.dataset)
        print(f'====> Epoch: {epoch} Average loss: 'f'{train_rcon_loss + train_prior_loss:.4f}')
        model.eval()
        val_rcon_loss, val_prior_loss = 0, 0
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                data = data.to(device)
                recon_batch, mu, logvar = model(data)
                rcon_loss, prior_loss = criterion(recon_batch, data, mu, logvar)
                val_rcon_loss += rcon_loss.item()
                val_prior_loss += prior_loss.item()
                if i == 0:
                    n = min(data.size(0), 8)
                    comparison = torch.cat([data[:n], recon_batch.view(batch_size, 1, *config['input_shape'])[:n]])
                    save_image(comparison.cpu(), f'{save_path}/{run_name}_reconstruction_{epoch}.png', nrow=n)
        val_rcon_loss /= len(val_loader.dataset)
        val_prior_loss /= len(val_loader.dataset)
        print(f'====> Validation set loss: {(val_rcon_loss + val_prior_loss):.4f}')
        log.step({'train': {'reconstruction_loss': train_rcon_loss, 'prior_loss': train_prior_loss, 'epoch': epoch},
                  'val': {'reconstruction_loss': val_rcon_loss, 'prior_loss': val_prior_loss, 'epoch': epoch}})
        if epoch % log_interval == 0:
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()},
                       f'{save_path}/{run_name}_checkpoint_{epoch}.pt')
    torch.save(model.state_dict(), f'{save_path}/{run_name}.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ukbb', help='dataset name')
    parser.add_argument('--model', type=str, default='icvae', help='model name')
    parser.add_argument('--cfg', type=str, default='default.yaml', help='config file')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--log_interval', type=int, default=10, help='log and save checkpoint interval')
    parser.add_argument('--sample_size', type=int, default=-1, help='number of samples to use')
    parser.add_argument('--val_size', type=float, default=0.2, help='validation size')
    parser.add_argument('--test_size', type=float, default=0.1, help='test size')
    parser.add_argument('--redo_splits', action='store_true', help='redo train/val/test splits')
    parser.add_argument('--device', type=str, default='cpu', help='device (cuda or cpu)')
    parser.add_argument('--save_path', type=str, default='checkpoints', help='save path')

    args = parser.parse_args()
    datapath = Path('datasets', args.dataset)
    config = load_yaml(Path('cfg', args.cfg))
    train_data, val_data, test_data = load_datasets(datapath, config['input_shape'], args.sample_size,
                                                    args.val_size, args.test_size, args.redo_splits,
                                                    shuffle=True, random_state=42)
    save_path = Path(args.save_path, args.dataset)
    if not save_path.exists():
        save_path.mkdir(parents=True)
    run_name = f'{args.model}_b{args.batch_size}_lr{args.lr * 1000:.0f}e-3_e{args.epochs}'
    train(args.model, config, train_data, val_data, args.batch_size, args.lr, args.epochs, args.log_interval,
          args.device, run_name, save_path)
