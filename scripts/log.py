import wandb
from os import environ


def init(project, name, latent_dim, lr, batch_size, epoch, sample_size, model, offline=False):
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
    wandb.watch(model)

    return run


def step(metrics_dict):
    wandb.log(metrics_dict)
