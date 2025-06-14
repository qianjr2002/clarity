import os
import logging
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from models.system import LiteSE
from data.cpc3_dataset import CPC3Dataset

torch.set_float32_matmul_precision('medium')  # or 'high' for better performance (slightly less precision)

log = logging.getLogger(__name__)

class LiteSELightningModule(LightningModule):
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
            # num_workers=self.cfg.training.num_workers,  # Use the same as train_dataloader
            collate_fn=CPC3Dataset.collate_fn
        )


@hydra.main(config_path="configs", config_name="model", version_base="1.3.2")
def main(cfg: DictConfig):
    seed_everything(cfg.training.seed)

    # output_dir = HydraConfig.get().runtime.output_dir
    # save_dir = os.path.join(output_dir, cfg.training.save_dir)
    # os.makedirs(save_dir, exist_ok=True)

    model_name = cfg.logging.model_name
    version = cfg.logging.version

    base_log_dir = os.path.join("logs", model_name)
    version_dir = os.path.join(base_log_dir, version)
    ckpt_dir = os.path.join(version_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    tb_logger = TensorBoardLogger(
        save_dir=base_log_dir,
        name=version,
        default_hp_metric=False
    )

    # checkpoint回调
    checkpoint_callback = ModelCheckpoint(
        monitor='Loss/val',
        dirpath=ckpt_dir,
        save_top_k=3,
        save_last=True,
        mode='min',
    )

    # early stopping
    early_stop_callback = EarlyStopping(
        monitor='Loss/val',
        patience=cfg.training.early_stop_patience,
        verbose=True,
        mode='min'
    )

    # 记录学习率
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    model = LiteSELightningModule(cfg)

    trainer = Trainer(
        max_epochs=cfg.training.epochs,
        logger=tb_logger,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        precision="16-mixed",  # mixed precision
        accelerator='auto',
        devices='auto',
        # strategy='ddp',     # 使用DDP多卡训练
        check_val_every_n_epoch=cfg.training.validate_every,
        log_every_n_steps=10,
        default_root_dir=base_log_dir,
    )

    trainer.fit(model,ckpt_path=cfg.training.get("resume_path", None))


if __name__ == "__main__":
    main()
