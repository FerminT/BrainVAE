from pathlib import Path
from scripts.data_handler import get_loader, load_datasets
from scripts.utils import load_yaml, load_architecture, save_reconstruction_batch
from scripts import log
from tqdm import tqdm
import argparse
import torch


def train(model_name, config, train_data, val_data, batch_size, lr, epochs, log_interval, device, no_sync, save_path):
    model, optimizer, criterion = load_architecture(model_name, config, device, lr)
    train_loader = get_loader(train_data, batch_size, shuffle=False)
    val_loader = get_loader(val_data, batch_size, shuffle=False)
    weights_path = save_path / 'weights'
    epoch, best_val_loss = log.resume(project='BrainVAE',
                                      run_name=f'{save_path.parent.name}_{save_path.name}',
                                      model=model,
                                      optimizer=optimizer,
                                      lr=lr,
                                      batch_size=batch_size,
                                      epochs=epochs,
                                      latent_dim=config['params']['latent_dim'],
                                      sample_size=len(train_data),
                                      weights_path=weights_path,
                                      offline=no_sync)
    while epoch < epochs:
        avg_rcon_loss, avg_prior_loss = train_epoch(model, train_loader, optimizer, criterion, log_interval, epoch)
        print(f'====> Epoch: {epoch} Avg loss: 'f'{avg_rcon_loss + avg_prior_loss:.4f}')
        val_rcon_loss, val_prior_loss = eval_epoch(model, val_loader, criterion, device, epoch, save_path)
        total_val_loss = val_rcon_loss + val_prior_loss
        print(f'====> Validation set loss: {total_val_loss:.4f}')
        log.step({'train': {'reconstruction_loss': avg_rcon_loss, 'prior_loss': avg_prior_loss, 'epoch': epoch},
                  'val': {'reconstruction_loss': val_rcon_loss, 'prior_loss': val_prior_loss, 'epoch': epoch}})
        log.save_ckpt(epoch, model.state_dict(), optimizer.state_dict(), total_val_loss, best_val_loss,
                      weights_path)
        best_val_loss = min(best_val_loss, total_val_loss)
        epoch += 1
    log.finish(model.state_dict(), save_path)


def train_epoch(model, train_loader, optimizer, criterion, log_interval, epoch):
    rcon_loss, prior_loss = 0, 0
    model.train()
    for batch_idx, (t1_imgs, ages) in enumerate(pbar := tqdm(train_loader)):
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(t1_imgs)
        recon_loss, prior_loss = criterion(recon_batch, t1_imgs, mu, logvar)
        loss = recon_loss + prior_loss
        loss.backward()
        rcon_loss += recon_loss.item()
        prior_loss += prior_loss.item()
        optimizer.step()
        pbar.set_description(f'Epoch {epoch} - loss: {loss.item():.4f}')
        if batch_idx % log_interval == 0:
            log.step({'train': {'batch': batch_idx, 'reconstruction_loss': recon_loss, 'prior_loss': prior_loss}})

    return rcon_loss / len(train_loader.dataset), prior_loss / len(train_loader.dataset)


def eval_epoch(model, val_loader, criterion, epoch, save_path):
    model.eval()
    val_rcon_loss, val_prior_loss = 0, 0
    with torch.no_grad():
        for i, (t1_imgs, ages) in enumerate(val_loader):
            t1_imgs = t1_imgs.to(device)
            recon_batch, mu, logvar = model(t1_imgs)
            rcon_loss, prior_loss = criterion(recon_batch, t1_imgs, mu, logvar)
            val_rcon_loss += rcon_loss.item()
            val_prior_loss += prior_loss.item()
            if i == 0:
                save_reconstruction_batch(t1_imgs, recon_batch, epoch, save_path)

    return val_rcon_loss / len(val_loader.dataset), val_prior_loss / len(val_loader.dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ukbb', help='dataset name')
    parser.add_argument('--model', type=str, default='vae', help='model name')
    parser.add_argument('--cfg', type=str, default='default.yaml', help='config file')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--log_interval', type=int, default=10, help='log and save checkpoint interval')
    parser.add_argument('--sample_size', type=int, default=-1, help='number of samples to use')
    parser.add_argument('--val_size', type=float, default=0.2, help='validation size')
    parser.add_argument('--test_size', type=float, default=0.1, help='test size')
    parser.add_argument('--redo_splits', action='store_true', help='redo train/val/test splits')
    parser.add_argument('--no_sync', action='store_true', help='do not sync to wandb')
    parser.add_argument('--device', type=str, default='cpu', help='device (cuda or cpu)')
    parser.add_argument('--save_path', type=str, default='checkpoints', help='save path')

    args = parser.parse_args()
    datapath = Path('datasets', args.dataset)
    config = load_yaml(Path('cfg', args.cfg))
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('cuda is not available')
    device = torch.device(args.device)
    train_data, val_data, test_data = load_datasets(datapath, config['params']['input_shape'], args.sample_size,
                                                    args.val_size, args.test_size, args.redo_splits, device,
                                                    shuffle=True, random_state=42)
    save_path = Path(args.save_path, args.dataset, args.model, args.cfg.split('.')[0])
    run_name = f'b{args.batch_size}_lr{args.lr * 1000:.0f}e-3_e{args.epochs}'
    save_path = save_path / run_name
    if not save_path.exists():
        save_path.mkdir(parents=True)
    train(args.model, config, train_data, val_data, args.batch_size, args.lr, args.epochs, args.log_interval,
          device, args.no_sync, save_path)
