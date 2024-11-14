#!/usr/bin/env python
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import lightning.pytorch as pl
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from src.utils.utils import set_seed
from src.models.lightning_models import SelfSupervisedModel
from src.data.in1k_data import SSLDataModule

@hydra.main(config_path="../configs/pretraining", config_name="pretrain_in1k_0.75", version_base="1.1")
def pretrain(cfg: DictConfig):

    print("Current working directory:", os.getcwd())
    print("Configuration:\n", OmegaConf.to_yaml(cfg))

    # Make deterministic
    set_seed(cfg.experiment.seed)

    data = SSLDataModule(
        train_path=cfg.data.train_path,
        val_path=cfg.data.val_path,
        img_size=cfg.data.img_size,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers
    )

    base_dir = "pretrain_outputs/"
    log_dir = os.path.join(base_dir, "csv_logs")
    os.makedirs(log_dir, exist_ok=True)

    # Set up the CSVLogger without specifying the version
    csv_logger = CSVLogger(save_dir=log_dir, name=cfg.experiment.name)

    print(f"Starting training {cfg.experiment.name} version {csv_logger.version} over {cfg.training.epochs} epochs")

    # Set up the checkpoint callback using the logger's version
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(base_dir, "checkpoints", cfg.experiment.name, f"version_{csv_logger.version}"),
        filename="{epoch:02d}-{val_loss:.3f}",
        save_top_k=-1,
        every_n_epochs=10,
        save_last=True,
        save_on_train_epoch_end=True,
        monitor="val_loss",
        mode="min",
    )

    # Set up the trainer
    trainer = pl.Trainer(
        strategy="ddp" if cfg.compute.devices > 1 else "auto",
        accelerator=cfg.compute.accelerator,
        devices=cfg.compute.devices,
        max_epochs=cfg.training.epochs,
        logger=csv_logger,
        callbacks=[checkpoint_callback],
        check_val_every_n_epoch=cfg.training.check_val_every_n_epoch,
        overfit_batches=cfg.training.overfit_batches,  # Use overfitting as a debugging technique
        num_sanity_val_steps=0 if cfg.training.overfit_batches > 0 else 2
    )

    model = SelfSupervisedModel(
        model_name=cfg.model.name,
        norm_pix_loss=cfg.model.norm_pix_loss,
        weight_decay=cfg.optimizer.weight_decay,
        beta1=cfg.optimizer.beta1,
        beta2=cfg.optimizer.beta2,
        epochs=cfg.training.epochs,
        learning_rate=cfg.training.learning_rate,
        optimizer=cfg.optimizer.name,
        warmup_epochs=cfg.scheduler.warmup_epochs,
        mask_ratio=cfg.mae.mask_ratio,
        batch_size_per_gpu=cfg.training.batch_size
    )

    # Handle checkpoint resumption
    ckpt_path = cfg.get("resume_from_checkpoint", None)
    if ckpt_path:
        if os.path.exists(ckpt_path):
            print(f"Resuming training from checkpoint: {ckpt_path}")
        else:
            raise FileNotFoundError(f"Checkpoint path '{ckpt_path}' does not exist.")
    else:
        print("No checkpoint provided. Starting training from scratch.")

    # Pass the checkpoint path to the trainer.fit method
    trainer.fit(model=model, datamodule=data, ckpt_path=ckpt_path)


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')

    pretrain()

