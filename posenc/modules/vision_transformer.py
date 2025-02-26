from typing import Tuple, Union

import lightning.pytorch as L
import torch
from monai.networks.nets import DenseNet121
from torch import nn

from posenc.enums import (
    ModelType,
    OptimizerType,
    PatchEmbeddingType,
    PosEncType,
    SchedulerType,
    ViTSettings,
)
from posenc.metrics import MultiLabelPerformance, get_classification_metrics
from posenc.nets.models import ViT, ViTLucid
from posenc.nets.optim import WarmupWithCosineDecay, WarmupWithExponentialDecay


def get_final_prediction(logits) -> torch.Tensor:
    """Get final prediction from model output."""
    return nn.functional.softmax(logits, dim=1).argmax(dim=1)


CLASS_NAMES = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Hernia",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pleural Thickening",
    "Pneumonia",
    "Pneumothorax",
    "Pneumoperitoneum",
    "Pneumomediastinum",
    "Subcutaneous Emphysema",
    "Tortuous Aorta",
    "Calcification of the Aorta",
    "No Finding",
]


class ViTMultiClsModule(L.LightningModule):
    def __init__(
        self,
        posenc: PosEncType,
        model_type: ModelType,
        optimizer: OptimizerType,
        lr: float,
        weight_decay: float,
        scheduler: SchedulerType,
        warmup_epochs: int,
        img_size: Union[int, Tuple[int, int]],
        patch_embedding: PatchEmbeddingType,
        scale: float = 1.0,
        temperature: int = 10000,
        variance_factors=None,
    ):
        super().__init__()

        self.num_classes = 20
        self.optimizer = optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler = scheduler
        self.warmup_epochs = warmup_epochs
        self.model_type = model_type
        self.scale = scale
        self.temperature = temperature
        self.ce_weights = [
            0.89,
            0.98,
            0.96,
            0.98,
            0.88,
            0.98,
            0.98,
            1.0,
            0.81,
            0.95,
            0.94,
            0.97,
            0.99,
            0.97,
            1.0,
            1.0,
            0.98,
            0.99,
            0.99,
            0.5,
        ]
        if model_type == ModelType.CNN:
            self.model = DenseNet121(
                spatial_dims=2,
                in_channels=1,
                out_channels=20,
                pretrained=False,
            )
        elif "vit" in model_type.value:
            vit_settings = ViTSettings(model_type.value)

            self.model = ViTLucid(
                posenc=posenc,
                patch_embed_type=patch_embedding,
                img_size=img_size,
                num_classes=20,
                scale=scale,
                temperature=temperature,
                mlp_dim=vit_settings.mlp_dim,
                num_layers=vit_settings.num_layers,
                num_heads=vit_settings.num_heads,
                variance_factors=variance_factors,
            )
        else:
            raise ValueError(f"Model type '{model_type}' not supported.")

        self.loss_function = nn.BCEWithLogitsLoss(weight=torch.tensor(self.ce_weights))

        # Define metrics
        self.train_metrics = MultiLabelPerformance(CLASS_NAMES, name_prefix="train/")
        self.valid_metrics = MultiLabelPerformance(CLASS_NAMES, name_prefix="valid/")

        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.model_type == ModelType.CNN:
            return self.model(x)

        return self.model(x)

    def _step(self, batch: Tuple[torch.Tensor, torch.Tensor]):
        """Common forward step for training and validation with loss computation."""
        images, labels = batch[0], batch[1]

        pred = self.forward(images)
        loss = self.loss_function(pred, labels)

        return loss, pred, labels

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        loss, pred, labels = self._step(batch)

        # Log metrics
        # Labels are soft labels because of MixUp and CutMix. Convert to hard labels.
        self.train_metrics(pred, labels)

        self.log("train/loss", loss, prog_bar=True)
        self.log_dict(self.train_metrics.compute(), on_step=False, on_epoch=True)

        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss, pred, labels = self._step(batch)

        # Log metrics
        self.valid_metrics(pred, labels)

        self.log("valid/loss", loss, prog_bar=True)
        self.log_dict(self.valid_metrics.compute(), on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        if self.optimizer == OptimizerType.ADAMW:
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        elif self.optimizer == OptimizerType.SGD:
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
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
