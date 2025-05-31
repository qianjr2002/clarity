import torch
import torch.nn as nn
from .model import Model
from pytorch_lightning import LightningModule as pl
from omegaconf import DictConfig
from data.cpc3_dataset import CPC3Dataset
from torch.utils.data import DataLoader


class BaseSE(pl):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg
        self.model = LiteSE(cfg.model)
        self.lr = cfg.training.lr

    def forward(self, signal, reference, train=True):
        return self.model(signal, reference, train=train)

    def training_step(self, batch, batch_idx):
        signals = [s.to(self.device) for s in batch["signals"]]
        references = [r.to(self.device) for r in batch["references"]]

        batch_loss = 0
        for signal, reference in zip(signals, references):
            min_length = min(signal.shape[-1], reference.shape[-1])
            signal = signal[..., :min_length]
            reference = reference[..., :min_length]
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                loss, _ = self.model(signal, reference, train=True)
            batch_loss += loss

        avg_batch_loss = batch_loss / len(signals)
        self.log('Loss/train', avg_batch_loss, on_step=False, on_epoch=True, prog_bar=True)
        return avg_batch_loss

    def validation_step(self, batch, batch_idx):
        signals = [s.to(self.device) for s in batch["signals"]]
        references = [r.to(self.device) for r in batch["references"]]

        batch_loss = 0
        for signal, reference in zip(signals, references):
            min_length = min(signal.shape[-1], reference.shape[-1])
            signal = signal[..., :min_length]
            reference = reference[..., :min_length]
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                loss, _ = self.model(signal, reference, train=True)
            batch_loss += loss.item()

        avg_batch_loss = batch_loss / len(signals)
        self.log('Loss/val', avg_batch_loss, on_step=False, on_epoch=True, prog_bar=True)
        return avg_batch_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=self.cfg.training.scheduler_factor,
            patience=self.cfg.training.scheduler_patience
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'Loss/val',
                'interval': 'epoch',
                'frequency': 1
            }
        }

    def train_dataloader(self):
        dataset = CPC3Dataset(
            data_root=self.cfg.data.root,
            split="train",
            target_sample_rate=self.cfg.audio.sample_rate,
            use_demo_data=self.cfg.data.use_demo
        )
        return DataLoader(
            dataset,
            batch_size=self.cfg.training.batch_size,
            shuffle=True,
            num_workers=self.cfg.training.num_workers,
            collate_fn=CPC3Dataset.collate_fn
        )

    def val_dataloader(self):
        dataset = CPC3Dataset(
            data_root=self.cfg.data.root,
            split="dev",
            target_sample_rate=self.cfg.audio.sample_rate,
            use_demo_data=self.cfg.data.use_demo
        )
        return DataLoader(
            dataset,
            batch_size=self.cfg.training.batch_size,
            shuffle=False,
            # num_workers=0,
            num_workers=self.cfg.training.num_workers,  # Use the same as train_dataloader
            collate_fn=CPC3Dataset.collate_fn
        )

from loss import BinauralLoss
class LiteSE(nn.Module):
    """Lightweight Speech Enhancement model for CPC3."""
    def __init__(self, cfg):
        super().__init__()
        self.model = Model()
        self.loss = BinauralLoss()

    def forward(self, noisy_wav, clean_wav=None, train=True):
        # noisy_wav.shape torch.Size([2, 176823])

        # TODO : padding
        noisy_wav = noisy_wav.unsqueeze(0)
        clean_wav = clean_wav.unsqueeze(0)
        est_wav = self.model(noisy_wav)

        min_length = min(est_wav.shape[-1], clean_wav.shape[-1])
        est_wav = est_wav[..., :min_length]
        clean_wav = clean_wav[..., :min_length]

        if train:
            loss = self.loss(est_wav, clean_wav)
            return loss, est_wav
        return est_wav
