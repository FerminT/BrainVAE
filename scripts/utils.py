import yaml
from torch import cat
from torchvision.utils import save_image


def save_reconstruction_batch(data, recon_batch, epoch, run_name, save_path):
    n, n_slice = min(data.size(0), 8), 50
    imgs_path = save_path / 'reconstructions'
    imgs_path.mkdir(exist_ok=True)
    for axis in range(3):
        if axis == 0:
            original_slice = data[:, :, n_slice, :, :]
            reconstructed_slice = recon_batch[:, :, n_slice, :, :]
        elif axis == 1:
            original_slice = data[:, :, :, n_slice, :]
            reconstructed_slice = recon_batch[:, :, :, n_slice, :]
        else:
            original_slice = data[:, :, :, :, n_slice]
            reconstructed_slice = recon_batch[:, :, :, :, n_slice]
        comparison = cat([original_slice[:n], reconstructed_slice[:n]])
        save_image(comparison.cpu(), imgs_path / f'{run_name}_reconstruction_{epoch}_axis_{axis}.png', nrow=n)


def load_yaml(filepath):
    with filepath.open('r') as file:
        return yaml.safe_load(file)
