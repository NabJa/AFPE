from typing import Tuple

import lightning.pytorch as L

from posenc.datasets.brats import BratsDatasetDataModule
from posenc.datasets.chestx import ChestObjDetDataModule, ChestXDataModule
from posenc.datasets.echonet import EchoNetDataModule, EchoNetDataModuleV2
from posenc.enums import (
    DataTaskType,
    ModelType,
    OptimizerType,
    PatchEmbeddingType,
    PosEncType,
    SchedulerType,
)
from posenc.modules.video_regression import VideoViTModule
from posenc.modules.vision_transformer import ViTBinaryClsModule, ViTMultiClsModule


def create_experiment(
    task: DataTaskType,
    model_type: ModelType = ModelType.VIT_B,
    posenc: PosEncType = PosEncType.SINCOS,
    batch_size: int = 32,
    num_workers: int = 4,
    optimizer: OptimizerType = OptimizerType.SGD,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    scheduler: SchedulerType = SchedulerType.WARMUPCOSINE,
    warmup_epochs: int = 10,
    scale: float = 1.0,
    temperature: int = 10000,
    variance_factors: Tuple[float, float] = None,
) -> Tuple[L.LightningDataModule, L.LightningModule]:

    if task == DataTaskType.CHESTX_MULTI:
        data_module = ChestXDataModule(
            "multilabel", batch_size, num_workers, do_cutmix=False, do_mixup=False
        )
        model = ViTMultiClsModule(
            posenc,
            model_type,
            optimizer,
            lr,
            weight_decay,
            scheduler,
            warmup_epochs,
            img_size=224,
            patch_embedding=PatchEmbeddingType.CONV,
            scale=scale,
            temperature=temperature,
            variance_factors=variance_factors,
        )


    elif task == DataTaskType.ECHONET_REG:
        n_frames = 16
        sampling_rate = 4
        data_module = EchoNetDataModuleV2(
            batch_size, num_workers, length=n_frames, period=sampling_rate
        )
        model = VideoViTModule(
            posenc=posenc,
            model_type=model_type,
            optimizer=optimizer,
            lr=lr,
            weight_decay=weight_decay,
            scheduler=scheduler,
            warmup_epochs=warmup_epochs,
            n_frames=n_frames,
            scale=scale,
            temperature=temperature,
            variance_factors=variance_factors,
        )
    else:
        raise NotImplementedError(f"DataModule {task} not implemented.")

    return data_module, model
