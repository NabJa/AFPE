import gc
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import optuna
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

import wandb
from posenc.enums import (
    DataTaskType,
    ModelType,
    OptimizerType,
    PosEncType,
    SchedulerType,
)
from posenc.experiment_factory import create_experiment

SEED = 4242

seed_everything(SEED)


@dataclass
class Defaults:
    task = DataTaskType.ECHONET_REG
    model = ModelType.VIT_SHALLOW
    positional_encoding = PosEncType.ISOFPE
    batch_size = 32
    num_workers = 32
    epochs = 75
    optimizer = OptimizerType.SGD
    scale = 1.0
    temperature = 10000
    warmup_epochs = 10
    weight_decay = 0.0
    precision = "32"
    lr = 0.001
    scheduler = SchedulerType.WARMUPEXP
    log_every_n_steps = 3


def objective(trial: optuna.Trial, dirpath: str) -> float:

    space_variance = trial.suggest_float("space_variance", 0.1, 1.0)
    time_variance = trial.suggest_float("time_variance", 0.1, 1.0)

    defaults = Defaults()

    # Load the datamodule and model
    datamodule, model = create_experiment(
        defaults.task,
        defaults.model,
        defaults.positional_encoding,
        defaults.batch_size,
        defaults.num_workers,
        defaults.optimizer,
        defaults.lr,
        defaults.weight_decay,
        defaults.scheduler,
        defaults.warmup_epochs,
        defaults.scale,
        defaults.temperature,
        variance_factors=[time_variance, space_variance, space_variance],
    )

    # Logging
    logger = WandbLogger(
        project=defaults.task.value,
        name=f"Trial-{trial.number:03}",
        save_dir="/sc-projects/sc-proj-gbm-radiomics/posenc/wandb",
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # Checkpoint at best validation loss
    checkpoint_path = Path(f"{dirpath}/Trial-{trial.number:03}")
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    checkpoint = ModelCheckpoint(
        dirpath=checkpoint_path,
        filename="{epoch:02d}",
        save_top_k=1,
        monitor="valid/loss",
        mode="min",
    )

    # Create a Lightning trainer
    trainer = Trainer(
        accelerator="gpu",
        max_epochs=defaults.epochs,
        precision=defaults.precision,
        gradient_clip_algorithm="norm",
        deterministic="warn",
        log_every_n_steps=defaults.log_every_n_steps,
        logger=logger,
        callbacks=[lr_monitor, checkpoint],
        check_val_every_n_epoch=2,
    )

    trainer.fit(model, datamodule)

    valid_r2 = trainer.callback_metrics["valid/r2"].item()

    wandb.finish()

    # Free memory
    del model
    del datamodule
    del trainer

    torch.cuda.empty_cache()
    gc.collect()

    return valid_r2


if __name__ == "__main__":

    root = "/sc-projects/sc-proj-gbm-radiomics/posenc/checkpoints/echonetreg/isofpe"

    study = optuna.create_study(
        study_name="EchoNetReg",
        storage=f"sqlite:///{root}/echonetreg.db",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(),  # Dont set seed! It would sample always the same hyperparams!
        direction="maximize",
    )

    obj = partial(objective, dirpath=root)
    study.optimize(obj, n_trials=100, gc_after_trial=True)
