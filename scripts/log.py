from scripts.utils import reconstruction_comparison_grid
from lightning.pytorch.callbacks import Callback


class LogReconstructionsCallback(Callback):
    def __init__(self, sample_size, slice_idx=50):
        self.sample_size = sample_size
        self.slice_idx = slice_idx

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx == 0:
            x, condition = batch
            n, n_slice = min(self.sample_size, x.size(0)), self.slice_idx
            imgs, captions = reconstruction_comparison_grid(x, outputs, n, n_slice, trainer.current_epoch)
            trainer.logger.log_image(key='reconstructions', images=imgs, caption=captions)
