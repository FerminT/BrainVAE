import wandb
from os import environ
from torch import save, load
from scripts.utils import save_reconstruction_batch
from lightning.pytorch.callbacks import Callback


class LogReconstructionsCallback(Callback):
    def __init__(self, sample_size, slice_idx=50):
        self.sample_size = sample_size
        self.slice_idx = slice_idx

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx == 0:
            x, condition = batch
            n, n_slice = min(self.sample_size, x.size(0)), self.slice_idx
            imgs, captions = save_reconstruction_batch(x, outputs, n, n_slice, trainer.current_epoch)
            trainer.logger.log_image(key='reconstructions', images=imgs, captions=captions)



def resume(project, run_name, model, optimizer, lr, batch_size, epochs, latent_dim, sample_size, weights_path,
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
        ckpt = load(weights_path / 'best.pt')
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        epoch, val_loss = ckpt['epoch'] + 1, ckpt['val_loss']
    else:
        epoch, val_loss = 0, float('inf')
    wandb.watch(model)
    return epoch, val_loss


def save_ckpt(epoch, model_state, optimizer_state, loss, is_best, weights_path):
    filename = weights_path / f'ckpt_{epoch}.pt'
    save_dict = {'epoch': epoch, 'model_state_dict': model_state, 'optimizer_state_dict': optimizer_state,
                 'val_loss': loss}
    save(save_dict, filename)
    if is_best:
        best_filename = weights_path / 'best.pt'
        save(save_dict, best_filename)


def step(metrics_dict, step, step_num):
    metrics_dict.update({step: step_num})
    wandb.log(metrics_dict)


def finish(model_state, weights_path):
    save(model_state, weights_path / 'final.pt')
    wandb.finish()
