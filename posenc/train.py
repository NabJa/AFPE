import argparse
import os

import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import SingleDeviceStrategy

from posenc.enums import (
    DataTaskType,
    ModelType,
    OptimizerType,
    PosEncType,
    SchedulerType,
)
from posenc.experiment_factory import create_experiment


def parse_args():
    parser = argparse.ArgumentParser()

    # Determine the task and model to use
    parser.add_argument("--task", type=DataTaskType, help="Name of the task to use")
    parser.add_argument(
        "--model", type=ModelType, default=ModelType.VIT_S, help="cnn or vit"
    )

    # This is what has to be changed for every run
    parser.add_argument(
        "--positional_encoding",
        type=PosEncType,
        default=PosEncType.SINCOS,
        help="Type of positional encoding to use",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for data loading"
    )
    parser.add_argument("--device", type=str, default="auto", help="Device to use.")

    # Everything from here should be the same for all tasks
    parser.add_argument(
        "--epochs", type=int, default=150, help="Number of epochs to train for"
    )
    parser.add_argument(
        "--optimizer",
        type=OptimizerType,
        default=OptimizerType.SGD,
        help="Optimizer to use",
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument(
        "--scheduler",
        type=SchedulerType,
        default=SchedulerType.WARMUPEXP,
        help="Scheduler to use",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Scale for fourier positional encoding.",
    )
    parser.add_argument(
        "--variance_factors",
        type=float,
        nargs="+",
        default=None,
        help="Variance factors for Anisotropic FPE.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=10000.0,
        help="Temperature for learnable positional encoding.",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=10,
        help="Number of warmup epochs. Only for scheduler with warmup.",
    )
    parser.add_argument(
        "--name_run", type=str, help="Name of the run for wandb logger."
    )
    parser.add_argument("--log_every_n_steps", type=int, default=10)
    parser.add_argument(
        "--overfit_single_batch",
        action="store_true",
        help="Enable overfit single batch for debugging",
    )
    parser.add_argument(
        "--seed", type=int, default=1234, help="Seed for reproducibility"
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="32",
        help="Options: 64, 32, 16, 16-mixed, bf16",
    )
    args = parser.parse_args()

    if args.num_workers is None:
        args.num_workers = max(os.cpu_count(), 64)

    if args.name_run is None:
        args.name_run = f"{args.model.value}_{args.positional_encoding.value}_{args.optimizer.value}_{args.scheduler.value}"

    if args.variance_factors is not None:
        if 3 < len(args.variance_factors) < 2:
            raise ValueError(
                "Variance factors should be a list of two or three floats."
            )

    return args


def main():
    # Parse command line arguments
    args = parse_args()

    seed_everything(args.seed)

    # Load the datamodule and model
    datamodule, model = create_experiment(
        args.task,
        args.model,
        args.positional_encoding,
        args.batch_size,
        args.num_workers,
        args.optimizer,
        args.lr,
        args.weight_decay,
        args.scheduler,
        args.warmup_epochs,
        args.scale,
        args.temperature,
        args.variance_factors,
    )

    # Do we do a debug run
    overfit_batches = 1 if args.overfit_single_batch else 0
    log_every_n_steps = 1 if args.overfit_single_batch else args.log_every_n_steps

    # Logging
    logger = WandbLogger(
        project=args.task.value,
        name=args.name_run,
        save_dir="/sc-projects/sc-proj-gbm-radiomics/posenc/wandb",
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # Checkpoint at best validation loss
    checkpoint = ModelCheckpoint(
        dirpath=f"/sc-projects/sc-proj-gbm-radiomics/posenc/checkpoints/{args.task.value}/{args.name_run}/{logger.experiment.id}",
        filename="{epoch:02d}",
        save_top_k=1,
        monitor="valid/loss",
        mode="min",
    )

    strategy = "auto"
    if args.device != "auto":
        # Hack for running multiple processes on the same node on SLURM.
        # The lightning SLURMStrategy does not work with multiple processes on the same node,
        # becuase it uses the same local_rank for all processes. The local_rank is ser by the
        # SLURM_LOCALID environment variable. We can set this variable manually to the device.
        os.environ["SLURM_LOCALID"] = str(args.device)

        # Set ddp if we are using multiple GPUs
        strategy = SingleDeviceStrategy(device=int(args.device))
        strategy.local_rank = int(args.device)

    # Create a Lightning trainer
    trainer = Trainer(
        accelerator="gpu",
        strategy=strategy,
        max_epochs=args.epochs,
        precision=args.precision,
        gradient_clip_algorithm="norm",
        deterministic="warn",
        log_every_n_steps=log_every_n_steps,
        logger=logger,
        callbacks=[lr_monitor, checkpoint],
        overfit_batches=overfit_batches,
        check_val_every_n_epoch=2,
    )

    trainer.fit(model, datamodule)


if __name__ == "__main__":
    # Settings
    torch.set_float32_matmul_precision("medium")

    main()
