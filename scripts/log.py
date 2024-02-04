import wandb
from os import environ


def init(project, name, latent_dim, lr, batch_size, epoch, sample_size, offline=False):
    if offline:
        environ['WANDB_MODE'] = 'offline'
    wandb.login()
    run = wandb.init(project=project, name=name, config={
        'latent_dim': latent_dim,
        'lr': lr,
        'batch_size': batch_size,
        'epoch': epoch,
        'sample_size': sample_size
    })

    return run


def step(train_loss, val_loss):
    wandb.log({'train_loss': train_loss, 'val_loss': val_loss})
