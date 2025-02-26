import gc
from functools import partial
from pathlib import Path

import optuna
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

import wandb
from posenc.datasets.chestx import ChestXDataModule
from posenc.enums import (
    ModelType,
    OptimizerType,
    PatchEmbeddingType,
    PosEncType,
    SchedulerType,
)
from posenc.modules.vision_transformer import ViTMultiClsModule

seed_everything(42)


def objective(trial: optuna.Trial, checkpoint_path, epochs) -> float:

    var_rows = trial.suggest_float("var_rows", 0.1, 2.0)
    var_cols = trial.suggest_float("var_cols", 0.1, 2.0)

    module = ViTMultiClsModule(
        posenc=PosEncType.ISOFPE,
        model_type=ModelType.VIT_B,
        optimizer=OptimizerType.ADAMW,
        lr=0.0001,
        weight_decay=0.01,
        scheduler=SchedulerType.WARMUPEXP,
        warmup_epochs=10,
        img_size=224,
        patch_embedding=PatchEmbeddingType.CONV,
        variance_factors=[var_rows, var_cols],
    )

    datamodule = ChestXDataModule(
        task="multilabel",
        batch_size=256,
        num_workers=32,
        do_cutmix=False,
        do_mixup=False,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    model_checkpoints = Path(checkpoint_path) / "models" / f"{int(trial.number):03}"
    model_checkpoints.mkdir(parents=True, exist_ok=True)
    checkpoints = ModelCheckpoint(
        dirpath=model_checkpoints,
        monitor="valid/loss",
        save_top_k=1,
    )

    logger = WandbLogger(
        save_dir=checkpoint_path,
        name=f"Trial-{trial.number:03}",
        project="ChestXFindFPE",
    )

    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=epochs,
        precision="bf16",
        gradient_clip_algorithm="norm",
        gradient_clip_val=2.0,
        deterministic="warn",
        log_every_n_steps=50,
        logger=logger,
        callbacks=[lr_monitor, checkpoints],
        check_val_every_n_epoch=2,
    )

    trainer.fit(model=module, datamodule=datamodule)

    final_loss = trainer.callback_metrics["valid/loss"].item()

    wandb.finish()

    # Free memory
    del module
    del datamodule
    del trainer

    torch.cuda.empty_cache()
    gc.collect()

    return final_loss


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")

    root = "/sc-scratch/sc-scratch-gbm-radiomics/posenc/models/ChestXFindFPE"

    objec = partial(
        objective,
        checkpoint_path=root,
        epochs=50,
    )

    study = optuna.create_study(
        study_name="ChestXFindFPE",
        storage=f"sqlite:///{root}/chestx_find_fpe.db",
        load_if_exists=True,
        sampler=optuna.samplers.RandomSampler(),
    )

    study.optimize(objec, n_trials=50, gc_after_trial=True)
