from typing import Tuple

import lightning.pytorch as L
import torch
import torch.nn as nn
import torchmetrics as tm
from monai.data import MetaTensor

from posenc.enums import (
    ModelType,
    OptimizerType,
    PosEncType,
    SchedulerType,
    ViTSettings,
)
from posenc.metrics import get_regression_metrics
from posenc.nets.models import VideoVisionTransformer
from posenc.nets.optim import WarmupWithCosineDecay, WarmupWithExponentialDecay


def maybe_as_tensor(x):
    if isinstance(x, MetaTensor):
        return x.as_tensor()
    return x


class EchonetRegMetrics:
    def __init__(self):
        self.metrics = {
            "mae": tm.MeanAbsoluteError(),
            "rmse": tm.MeanSquaredError(squared=False),
            "r2": tm.R2Score(),
        }

    def __call__(self, pred, label):
        return {name: f(pred, label).item() for name, f in self.metrics.items()}


class VideoViTModule(L.LightningModule):
    def __init__(
        self,
        posenc: PosEncType,
        model_type: ModelType,
        optimizer: OptimizerType,
        lr: float,
        weight_decay: float,
        scheduler: SchedulerType,
        warmup_epochs: int,
        n_frames: int = 16,
        scale: float = 1.0,
        temperature: int = 10000,
        variance_factors: Tuple[float, float] = None,
    ):
        super().__init__()

        self.scheduler = scheduler
        self.warmup_epochs = warmup_epochs
        self.weight_decay = weight_decay
        self.model_type = model_type
        self.optimizer = optimizer
        self.scale = scale
        self.temperature = temperature

        if model_type == ModelType.CNN:
            raise ValueError("CNN not supported for video generation!")

        vit_settings = ViTSettings(model_type.value)
        self.model = VideoVisionTransformer(
            posenc_type=posenc,
            image_size=112,
            patch_size_spatial=8,
            patch_size_temporal=2,
            num_frames=n_frames,
            num_layers=8,
            num_heads=8,
            hidden_dim=384,
            mlp_dim=1024,
            dropout=0.0,
            attention_dropout=0.0,
            temperature=temperature,
            scale=scale,
            variance_factors=variance_factors,
        )

        self.loss = nn.MSELoss()
        self.lr = lr

        # Metric trackers
        self.train_metric = get_regression_metrics("train")
        self.valid_metric = get_regression_metrics("valid")

        self.save_hyperparameters()

    def forward(self, x):
        # Ejection fraction is between 0 and 100
        y_hat = self.model(x).flatten().clip(0, 100)
        y_hat = maybe_as_tensor(y_hat)
        return y_hat

    def _step(self, batch):
        x = batch["image"]
        y = batch["target"]

        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)

        return loss, y_hat, y

    def training_step(self, batch, batch_idx):
        loss, y_hat, y = self._step(batch)

        self.train_metric(y_hat, y)

        self.log_dict(self.train_metric, on_epoch=True, on_step=False)
        self.log("train/loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_hat, y = self._step(batch)

        self.valid_metric(y_hat, y)

        self.log_dict(self.valid_metric, on_epoch=True, on_step=False)
        self.log("valid/loss", loss, prog_bar=True)

        return loss

    def configure_optimizers(self):
        if self.optimizer == OptimizerType.ADAMW:
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        elif self.optimizer == OptimizerType.SGD:
            optimizer = torch.optim.SGD(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"Invalid optimizer {self.optimizer}")

        if self.scheduler == SchedulerType.COSINE:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=10, eta_min=10e-8
            )
        elif self.scheduler == SchedulerType.WARMUPCOSINE:
            scheduler = WarmupWithCosineDecay(
                optimizer, warmup_steps=self.warmup_epochs, eta_min=10e-8
            )
        elif self.scheduler == SchedulerType.WARMUPEXP:
            scheduler = WarmupWithExponentialDecay(
                optimizer, gamma=0.97, warmup_steps=self.warmup_epochs, eta_min=10e-8
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
            },
        }
