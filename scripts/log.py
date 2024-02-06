import wandb
from os import environ
from torch import save, load


def resume(project, run_name, model, optimizer, lr, batch_size, epochs, latent_dim, sample_size, save_path,
           offline=False):
    if offline:
        environ['WANDB_MODE'] = 'offline'
    wandb.init(project=project, name=run_name, config={
        'latent_dim': latent_dim,
        'lr': lr,
        'batch_size': batch_size,
        'epochs': epochs,
        'sample_size': sample_size
    }, resume=True)
    if wandb.run.resumed:
        ckpt = load(save_path / f'{run_name}_best.pt')
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        epoch, val_loss = ckpt['epoch'] + 1, ckpt['val_loss']
    else:
        epoch, val_loss = 0, float('inf')
    wandb.watch(model)
    return epoch, val_loss


def save_ckpt(epoch, model_state, optimizer_state, loss, best_loss, save_path, run_name):
    filename = save_path / f'{run_name}_ckpt_{epoch}.pt'
    save_dict = {'epoch': epoch, 'model_state_dict': model_state, 'optimizer_state_dict': optimizer_state,
                 'val_loss': loss}
    save(save_dict, filename)
    if loss < best_loss:
        best_filename = save_path / f'{run_name}_best.pt'
        save(save_dict, best_filename)


def step(metrics_dict):
    wandb.log(metrics_dict)


def finish(model_state, save_path, run_name):
    save(model_state, save_path / f'{run_name}.pt')
    wandb.finish()
