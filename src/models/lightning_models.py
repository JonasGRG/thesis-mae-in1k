import math

import torch
import lightning.pytorch as pl
from torch.optim import Adam, AdamW
from timm.models.layers import trunc_normal_

from src.models import mae_vit, vit
from src.utils.lr_scheduler import get_cosine_schedule_with_warmup
from src.utils.positional_embedding import interpolate_pos_embed

class SelfSupervisedModel(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        norm_pix_loss: bool = False,
        weight_decay: float = 0.05,
        beta1: float = 0.9,
        beta2: float = 0.95,
        epochs: int = 100,
        learning_rate: float = 1.5e-4,
        optimizer: str = "adamw",
        warmup_epochs: int = 40,
        mask_ratio: float = 0.75,
        batch_size_per_gpu: int = 1024
    ):
        super().__init__()

        # Loss, optimizer and scheduler parameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer = optimizer
        self.beta1 = beta1
        self.beta2 = beta2
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.batch_size_per_gpu = batch_size_per_gpu

        self.mask_ratio = mask_ratio

        # Variables to be set in on_fit_start()
        self.num_gpus = None
        self.total_steps = None
        self.warmup_steps = None

        # Save params
        self.save_hyperparameters()  
        
        # Model 
        self.model = mae_vit.__dict__[model_name](norm_pix_loss=norm_pix_loss)

    def training_step(self, batch):
        samples, _ = batch
        loss, _, _ = self.model(samples, mask_ratio=self.mask_ratio)
        self.log("train_loss", loss, on_step=True, prog_bar=True, on_epoch=True, sync_dist=True)
        # Log the learning rate
        lr = self.optimizers().param_groups[0]['lr']
        self.log('learning_rate', lr, on_step=True, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch):
        samples, _ = batch
        loss, _, _ = self.model(samples, mask_ratio=self.mask_ratio)
        self.log("val_loss", loss, on_step=False, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        assert self.optimizer in ["adam", "adamw"]

        if self.optimizer == "adamw":
            optimizer = AdamW(
                self.parameters(), 
                lr=self.learning_rate,
                betas=(self.beta1, self.beta2), 
                weight_decay=self.weight_decay
                )
        elif self.optimizer == "adam":
            optimizer = Adam(
                self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        print(f"Using optimizer {self.optimizer}")

        # Ensure on_fit_start() has been called
        if self.total_steps is None or self.warmup_steps is None:
            self.on_fit_start()

        # Configure scheduler with correct steps
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.total_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step", # Ensure that scheduler is updated each step such that lr is not 0 for the entire first epoch
                "frequency": 1,
            },
        }
    
    def on_fit_start(self):
        # This method is called when fit begins
        # Get number of GPUs (devices)
        self.num_gpus = max(1, self.trainer.num_devices)

        # Get the total number of samples
        train_dataloader = self.trainer.datamodule.train_dataloader()
        train_dataset = train_dataloader.dataset

        # If using a DistributedSampler, get the underlying dataset
        if isinstance(train_dataset, torch.utils.data.Subset):
            total_num_samples = len(train_dataset.dataset)
        else:
            total_num_samples = len(train_dataset)

        # Compute total batch size and steps per epoch
        total_batch_size = self.batch_size_per_gpu * self.num_gpus
        steps_per_epoch = math.ceil(total_num_samples / total_batch_size)

        # Calculate total steps and warmup steps
        self.total_steps = self.epochs * steps_per_epoch
        self.warmup_steps = self.warmup_epochs * steps_per_epoch