import math

from torch.optim.lr_scheduler import _LRScheduler


class WarmupWithExponentialDecay(_LRScheduler):
    """Exponential decay learning rate scheduler with warmup.
    The defaults give a good LR curve for training with 150 epochs.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        gamma (float): Multiplicative factor of learning rate decay.
        warmup_steps (int): The number of warmup steps.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(
        self,
        optimizer,
        gamma=0.97,
        warmup_steps=10,
        eta_min=10e-8,
        last_epoch=-1,
    ):
        self.gamma = gamma
        self.eta_min = eta_min
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):

        if self.last_epoch <= self.warmup_steps:
            warum_reduction = self.last_epoch / self.warmup_steps
            return [
                max(base_lr * warum_reduction, self.eta_min)
                for base_lr in self.base_lrs
            ]

        return [
            max(group["lr"] * self.gamma, self.eta_min)
            for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self):
        return [base_lr * self.gamma**self.last_epoch for base_lr in self.base_lrs]


class WarmupWithCosineDecay(_LRScheduler):
    """Warmup with Cosine Decay learning rate scheduler."""

    def __init__(
        self,
        optimizer,
        warmup_steps,
        decay_factor=0.2,
        decay_period=500,
        eta_min=0,
        last_epoch=-1,
    ):
        """
        Initializes a custom optimizer scheduler.

        Args:
            optimizer (torch.optim.Optimizer): The base optimizer.
            warmup_steps (int): The number of warmup steps.
            decay_factor (float): The decay factor for the learning rate.
            decay_period (int, optional): The number of iterations for one cosine decay period restart. Defaults to 1.
            eta_min (float, optional): The minimum learning rate. Defaults to 0.
            last_epoch (int, optional): The index of the last epoch. Defaults to -1.
        """
        self.warmup_steps = warmup_steps
        self.decay_factor = decay_factor
        self.decay_period = decay_period
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [
                base_lr * (self.last_epoch / self.warmup_steps)
                for base_lr in self.base_lrs
            ]
        else:
            base_lrs = [
                base_lr
                * (
                    self.decay_factor
                    ** ((self.last_epoch - self.warmup_steps) // self.decay_period)
                )
                for base_lr in self.base_lrs
            ]
            return [
                lr
                * 0.5
                * (
                    1
                    + math.cos(
                        math.pi
                        * ((self.last_epoch - self.warmup_steps) % self.decay_period)
                        / self.decay_period
                    )
                )
                for lr in base_lrs
            ]
