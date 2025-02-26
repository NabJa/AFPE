import argparse
import multiprocessing
import os
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Callable, Optional, Tuple
from urllib import request

import lightning.pytorch as L
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.transforms import v2
from tqdm import tqdm

from posenc.datasets.transforms import (
    AddGaussianNoise,
    get_cutmix_and_mixup_collate_function,
)

# TODO: Change this to the correct path
ROOT = "/path/to/chestx"

# URLs for the zip files
IMAGE_LINKS = {
    "01": "https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz",
    "02": "https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz",
    "03": "https://nihcc.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20z2jp.gz",
    "04": "https://nihcc.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz",
    "05": "https://nihcc.box.com/shared/static/v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz",
    "06": "https://nihcc.box.com/shared/static/asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz",
    "07": "https://nihcc.box.com/shared/static/jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz",
    "08": "https://nihcc.box.com/shared/static/tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz",
    "09": "https://nihcc.box.com/shared/static/upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz",
    "10": "https://nihcc.box.com/shared/static/l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz",
    "11": "https://nihcc.box.com/shared/static/hhq8fkdgvcari67vfhs7ppg2w6ni4jze.gz",
    "12": "https://nihcc.box.com/shared/static/ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz",
}


########################
#### Preprocessing #####
########################

def norm_pixels(x, *args, **kwargs):
    """Normalize the intensity to 0 and 1."""
    return x / 255


def norm_intensity(x, *args, **kwargs):
    """Normalize the intensity of the image with dataset mean and std."""
    return (x - 0.486) / 0.246


def add_channel_dim(x, *args, **kwargs):
    """Add channel dimension to the image."""
    return x[None, ...]


def to_tensor_image(x, *args, **kwargs):
    """Add channel dimension to the image."""
    return torch.tensor(x[None, ...], dtype=torch.float32)


def _download_image(output_path: Path, link_key: str) -> None:
    """
    Download a single image from the NIH website
    """
    filepath = output_path / Path(f"images_{link_key}.tar.gz")
    request.urlretrieve(IMAGE_LINKS[link_key], filepath)


def download_images(output_path: Path, processes: int = 12):
    """
    Download the images from the NIH website and save them to the output path
    """
    _download = partial(_download_image, Path(output_path))

    with Pool(processes) as p:
        p.map(_download, list(IMAGE_LINKS.keys()))


def resize_image(image_path, output_path, new_size):
    """
    Resize a single image and save it to the specified path.

    Args:
        image_path: Path to the input image.
        output_path: Path where resized image will be saved.
        new_size: Tuple specifying the new size (width, height) for the image.
    """
    # Load image using PIL
    image = Image.open(image_path)

    # Define transformation to resize image
    transform = v2.Resize(new_size)

    # Apply transformation to resize image
    resized_image = transform(image)

    # Get original file name
    _, filename = os.path.split(image_path)

    # Save resized image with the same original file name
    resized_image_path = os.path.join(output_path, filename)
    resized_image.save(resized_image_path)


def resize_images(input_path, output_path, new_size, num_processes=24):
    """
    Resize all images in a directory and save them to a specified path.

    Args:
        input_path: Path to the directory containing input images.
        output_path: Path where resized images will be saved.
        new_size: Tuple specifying the new size (width, height) for the images.
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    input_path = Path(input_path)
    images = list(input_path.glob("*.png"))

    # Set up multiprocessing pool
    pool = multiprocessing.Pool(processes=num_processes)

    _resize = partial(resize_image, output_path=output_path, new_size=new_size)

    # Resize images in parallel
    for _ in tqdm(
        pool.imap_unordered(_resize, images, chunksize=10), total=len(images)
    ):
        pass

    # Close the pool to free resources
    pool.close()
    pool.join()


########################
##### Dataclasses ######
########################

class ChestXDataset:
    def __init__(
        self,
        mode="train",
        task="multilabel",
        image_path="resized256",
        transforms: Optional[Callable] = None,
        csv_root: Optional[str] = None,
    ):

        assert task in ["binary", "multilabel"], f"Task {task} not supported."

        self.root = Path(ROOT)
        self.transforms = transforms
        self.task = task
        self.image_path = image_path

        assert mode in [
            "train",
            "val",
            "test",
        ], "mode should be one of 'train', 'val', or 'test'"
        self.mode = mode

        if csv_root:
            self.csv_root = Path(csv_root)
        else:
            self.csv_root = self.root / "PruneCXR"

        self.csv_path = self.csv_root / f"miccai2023_nih-cxr-lt_labels_{mode}.csv"

        self.data = pd.read_csv(self.csv_path)

        if "subj_id" in self.data:
            self.data.drop(["subj_id"], axis=1, inplace=True)

        self.data.reset_index(drop=True, inplace=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        image_name = self.data.loc[idx, "id"]
        img = str(self.root / "images" / self.image_path / image_name)
        img = Image.open(img).convert("L")

        if self.task == "binary":
            label = self.data.loc[idx, "No Finding"]
        else:
            label = self.data.loc[
                idx,
                ~self.data.columns.isin(
                    ["id", "sampling_weight", "binary_sampling_weight"]
                ),
            ].values.astype(int)

        if self.transforms:
            img = self.transforms(img)

        return img, torch.tensor(label).float()

    @property
    def labels(self):
        return list(self.data.columns.drop("id"))


class ChestXDataModule(L.LightningDataModule):
    def __init__(
        self,
        task: str,
        batch_size: int = 64,
        num_workers: int = 12,
        image_size: int = 224,
        image_dir_name: str = "resized256",
        do_cutmix: bool = True,
        do_mixup: bool = True,
        csv_root: Optional[str] = None,
    ):
        """
        Args:
            dataset (str): Dataset to use. Options: binary | multilabel | detection
            batch_size (int, optional): Batch size. Defaults to 64.
            num_workers (int, optional): Number of CPU workers. Defaults to 12.
            train_transform (callable, optional): Transforms for train set. Defaults to None.
            valid_transform (callable, optional): Transforms for test and validation sets. Defaults to None.
            image_size (int, optional): Image size. Defaults to 224 as in original ViT. This is the size of the image after resizing.
            image_dir_name (str, optional): Image folder name to the images. Defaults to "resized256".
            csv_root (str, optional): Path to the root directory containing the CSV files. Defaults to None.
        """
        super().__init__()

        if task == "multilabel":
            self.num_classes = 20
            self.sampling_weights_key = (
                "binary_sampling_weight"  # TODO: Have to be computed.
            )
        elif task == "binary":
            self.num_classes = 2
            self.sampling_weights_key = "binary_sampling_weight"
        else:
            raise NotImplementedError(
                f"Only 'multilabel' and 'binary' are supported. Given {task}"
            )

        self.image_dir_name = image_dir_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.task = task
        self.image_size = image_size
        self.do_cutmix = do_cutmix
        self.do_mixup = do_mixup
        self.csv_root = csv_root

        # Define transforms
        self.train_transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.526], std=[0.252]),
                v2.RandomCrop(image_size),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomRotation(degrees=15),
                AddGaussianNoise(p=0.33, mean=0.0, std=0.1),
            ]
        )
        self.valid_transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.526], std=[0.252]),
                v2.RandomCrop(image_size),
            ]
        )

        self.save_hyperparameters()

    def setup(self, stage: str = None):
        self.train = ChestXDataset(
            mode="train",
            task=self.task,
            image_path=self.image_dir_name,
            transforms=self.train_transform,
            csv_root=self.csv_root,
        )
        self.valid = ChestXDataset(
            mode="val",
            task=self.task,
            image_path=self.image_dir_name,
            transforms=self.valid_transform,
            csv_root=self.csv_root,
        )
        self.test = ChestXDataset(
            mode="test",
            task=self.task,
            image_path=self.image_dir_name,
            transforms=self.valid_transform,
            csv_root=self.csv_root,
        )

    def train_dataloader(self):
        # This sampler is used to balance the classes.
        # It samples with replacement to make ensure minority classes are oversampled.
        sampler = WeightedRandomSampler(
            self.train.data[self.sampling_weights_key],
            len(self.train.data),
            replacement=True,
        )

        # Add this transform to collate to apply cutmix or mixup with multi-processing.
        collate_fn = get_cutmix_and_mixup_collate_function(
            self.do_cutmix, self.do_mixup, self.num_classes
        )

        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            drop_last=True,
            sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test, batch_size=self.batch_size, num_workers=self.num_workers
        )




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output_path", type=str, help="Output path for preprocessing or downloading"
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=12,
        help="Number of processes for downloading",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    download_images(Path(args.output_path), args.num_processes)
