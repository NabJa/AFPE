import numpy as np
import torch
from torch.utils.data.dataloader import default_collate
from torchvision.transforms import v2


class AddGaussianNoise:
    def __init__(self, p=0.0, mean=0.0, std=0.1):
        self.std = std
        self.mean = mean
        self.p = p

    def __call__(self, img):
        add_noise = np.random.binomial(1, self.p)
        if add_noise:
            return img + torch.randn(img.size()) * self.std + self.mean
        return img

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(
            self.mean, self.std
        )


def get_cutmix_and_mixup_collate_function(
    do_cutmix: bool, do_mixup: bool, num_classes: int
):
    if do_cutmix and do_mixup:
        cutmix_or_mixup = v2.RandomChoice(
            [
                v2.CutMix(num_classes=num_classes),
                v2.MixUp(num_classes=num_classes),
            ]
        )

        def collate_fn(batch):
            return cutmix_or_mixup(*default_collate(batch))

    elif do_cutmix:
        cutmix = v2.CutMix(num_classes=num_classes)

        def collate_fn(batch):
            return cutmix(*default_collate(batch))

    elif do_mixup:
        mixup = v2.MixUp(num_classes=num_classes)

        def collate_fn(batch):
            return mixup(*default_collate(batch))

    else:
        collate_fn = default_collate

    return collate_fn
