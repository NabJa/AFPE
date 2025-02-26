import collections
import os
from multiprocessing import Pool
from pathlib import Path
from typing import Optional

import cv2
import imageio
import lightning.pytorch as L
import monai.transforms as T
import numpy as np
import pandas
import pandas as pd
import skimage.draw
import torch
import torchvision
from einops import rearrange
from monai.data import DataLoader
from torchvision.transforms.functional import rgb_to_grayscale
from tqdm import tqdm

# TODO: Change this to the correct path
ROOT = "/path/to/echonet-dynamic"

# Used for normalization of the EchoNet dataset
ECHONET_STATS = {"mean": 57.6, "std": 54.1, "median": 40}
MEAN = np.array([32.756416, 32.941185, 33.275223])
STD = np.array([49.92367, 50.0535, 50.396824])

########################
#### Preprocessing #####
########################


def read_video(path):
    """Read video from path and return as numpy array with shape (frames, channels, height, width)."""
    vid = imageio.get_reader(path, "ffmpeg")
    video = [i for i in vid.iter_data()]
    return rearrange(
        video, "frames height width channels -> frames channels height width"
    )


def _save_transformed_video(path):
    video = torch.tensor(read_video(path))
    video = rgb_to_grayscale(video)
    torch.save(video, path.with_suffix(".pt"))


def transform_all_videos_to_grayscaled_tensors(input_dir, nprocesses=12):
    input_dir = Path(input_dir)
    videos = list(input_dir.glob("*.avi"))
    with Pool(nprocesses) as p:
        _ = list(p.imap(_save_transformed_video, tqdm(videos)))


########################
##### Transforms #######
########################


class ImageDropout:
    """Set a random channel and / or pixels to zero. This is for a image generation task."""

    def __init__(
        self,
        keys=None,
        channel_wise: bool = True,
        pixel_wise: bool = False,
        p: float = 0.5,
    ):
        """
        Args:
            channel_wise (bool, optional): Whether dropout channel-wise. Defaults to True.
            pixel_wise (bool, optional): Whether dropout pixel-wise. Defaults to False.
            p (float, optional): Probability of applying dropout. Must be in the range [0, 1]. Defaults to 0.5.

        Raises:
            AssertionError: If both channel_wise and pixel_wise are False.
            AssertionError: If p is not in the range [0, 1].
        """
        self.keys = keys
        self.channel_wise = channel_wise
        self.pixel_wise = pixel_wise
        self.p = p

        assert (
            channel_wise or pixel_wise
        ), "At least one of channel_wise or pixel_wise must be True."

        assert 0 < p < 1, "p must be in the range [0, 1]"

    def __call__(self, data: dict) -> torch.Tensor:
        """Set a random channel or values to zero. This is for a image generation task."""

        img = data[self.keys]

        if self.channel_wise:
            # Drop every channel with probability p
            mask = torch.rand(img.shape[0]) < self.p
            img[mask] = 0.0
            data["channel_dropped_idx"] = mask

        if self.pixel_wise:
            # Drop every foreground pixel with probability p
            fg_idx = torch.where(img > 0)
            fg_mask = torch.rand(fg_idx[0].shape) < self.p
            fg_idx = [i[fg_mask] for i in fg_idx]
            img[fg_idx] = 0

        data[self.keys] = img
        return data


class LoadVideoClip:
    """Read video and select random clip of length 'clip_length'."""

    def __init__(
        self, keys="image", clip_length: Optional[int] = 16, sampling_rate: int = 4
    ):
        """
        Args:
            keys (str): The keys to be loaded from the dataset. Defaults to "image".
            clip_length (int, optional): The length of each video clip. Defaults to 16. If None, the whole video is returned.
            sampling_rate (int, optional): The sampling rate for the video. Defaults to 1.
        """
        self.keys = keys
        self.clip_length = clip_length
        self.sampling_rate = sampling_rate

    def __call__(self, data: dict) -> torch.Tensor:
        """Read video and select random clip of length 'clip_length'."""

        path = data[self.keys]
        data["path"] = path

        video: torch.Tensor = torch.load(path, weights_only=False)
        video = video.float()

        if self.clip_length is None:
            data[self.keys] = video
            return data

        # Apply sampling rate
        if len(video) / self.sampling_rate > self.clip_length:
            video = video[:: self.sampling_rate]
        start = np.random.randint(0, len(video) - self.clip_length)
        data[self.keys] = video[start : start + self.clip_length]

        return data


########################
##### Dataclasses ######
########################


class EchoNetDataset:
    """
    The dataset is a collection of echocardiograms from the EchoNet challenge.
    It can be used to predict the ejection fraction (EF), end-diastolic volume (EDV), or end-systolic volume (ESV).
    We also use it for image generation tasks using the dropout transforms.
    """

    def __init__(
        self,
        mode: str = "train",
        output_variable: str = "ef",
        transforms=None,
    ) -> None:
        """
        Args:
            mode: The mode of the dataset. Can be "train", "val", or "test". Defaults to "train".
            output_variable: The output variable to predict. Can be "ef", "edv", or "esv". Defaults to "ef".
            transforms: A list of transforms to apply to the data. Defaults to None.
        """
        self.root = Path("/sc-scratch/sc-scratch-gbm-radiomics/posenc/echonet-dynamic")
        self.video_path = self.root / "processed"
        self.data = pd.read_csv(self.root / "FileList.csv")
        # self.volume_tracing = pd.read_csv(self.root / "VolumeTracings.csv")

        self.transforms = transforms

        assert output_variable.lower() in ["ef", "edv", "esv"]
        self.output_variable = output_variable.upper()

        assert mode in ["train", "val", "test"]
        self.mode = mode
        self.data = self.data[self.data.Split.str.lower() == mode]

        # Remove videos with FrameWidth or FrameHeight unequal 112
        self.data = self.data[
            (self.data.FrameWidth == 112) & (self.data.FrameHeight == 112)
        ]

        self.data.reset_index(inplace=True, drop=True)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        """Return a sample from the dataset."""

        file_name, out_variable = self.data.loc[idx, ["FileName", self.output_variable]]
        video_path = str(self.video_path / f"{file_name}.pt")
        target = torch.tensor(out_variable, dtype=torch.float32)

        sample = {
            "image": video_path,
            "target": target,
            "target_name": self.output_variable,
        }

        if self.transforms:
            sample = self.transforms(sample)

        return sample


class EchoNetDataModule(L.LightningDataModule):
    """
    DataModule for the EchoNet dataset.
    This class is used to load the data, create the dataloaders and handle the specific taks transformations.
    Can be regression of output_variable or image generation with channel or pixel wise dropout.
    """

    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 12,
        output_variable: str = "ef",
        clip_length: int = 16,
        sampling_rate: int = 1,
        dropout_channel: bool = False,
        dropout_pixel: bool = False,
        dropout_p: float = 0.5,
    ):
        """
        Initializes the Echonet dataset.

        Args:
            batch_size (int): The batch size for data loading. Default is 32.
            num_workers (int): The number of worker threads for data loading. Default is 12.
            output_variable (str): The output variable to predict. Default is "ef".
            clip_length (int): The length of video clips. Default is 16.
            dropout_channel (bool): Whether to apply channel dropout. Default is False.
            dropout_pixel (bool): Whether to apply pixel dropout. Default is False.
            dropout_p (float): The dropout probability. Default is 0.5.
        """
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.output_variable = output_variable
        self.clip_length = clip_length
        self.drop_channel = dropout_channel
        self.drop_pixel = dropout_pixel
        self.dropout_p = dropout_p
        self.sampling_rate = sampling_rate

        dropout = dropout_channel or dropout_pixel

        self.train_transforms = self._get_transforms(augment=True, dropout=dropout)
        self.valid_transform = self._get_transforms(augment=False, dropout=dropout)

        self.save_hyperparameters()

    def _get_transforms(self, augment: bool = False, dropout=False):
        transforms = [
            LoadVideoClip(
                clip_length=self.clip_length, sampling_rate=self.sampling_rate
            ),
            T.NormalizeIntensityd(
                keys="image",
                subtrahend=ECHONET_STATS["mean"],
                divisor=ECHONET_STATS["std"],
                nonzero=True,
            ),
        ]

        if dropout:
            transforms += [
                T.CopyItemsd(keys="image"),
                ImageDropout(
                    keys="image",
                    channel_wise=self.drop_channel,
                    pixel_wise=self.drop_pixel,
                    p=self.dropout_p,
                ),
            ]

        if augment:
            transforms += [
                T.RandGaussianNoised(keys="image", prob=0.33, mean=0, std=0.1),
                T.RandShiftIntensityd(keys="image", prob=0.33, offsets=0.1),
                T.RandScaleIntensityd(keys="image", prob=0.33, factors=0.1),
            ]

        return T.Compose(transforms)

    def setup(self, stage: str = None):
        self.train = EchoNetDataset(
            mode="train",
            output_variable=self.output_variable,
            transforms=self.train_transforms,
        )
        self.val = EchoNetDataset(
            mode="val",
            output_variable=self.output_variable,
            transforms=self.valid_transform,
        )
        self.test = EchoNetDataset(
            mode="test",
            output_variable=self.output_variable,
            transforms=self.valid_transform,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test, batch_size=self.batch_size, num_workers=self.num_workers
        )


###########################
##### Dataclasses V2 ######
###########################


def loadvideo(filename: str) -> np.ndarray:
    """Loads a video from a file.

    Args:
        filename (str): filename of video

    Returns:
        A np.ndarray with dimensions (channels=3, frames, height, width). The
        values will be uint8's ranging from 0 to 255.

    Raises:
        FileNotFoundError: Could not find `filename`
        ValueError: An error occurred while reading the video
    """

    if not os.path.exists(filename):
        raise FileNotFoundError(filename)
    capture = cv2.VideoCapture(filename)

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    v = np.zeros((frame_count, frame_height, frame_width, 3), np.uint8)

    for count in range(frame_count):
        ret, frame = capture.read()
        if not ret:
            raise ValueError("Failed to load frame #{} of {}.".format(count, filename))

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        v[count, :, :] = frame

    v = v.transpose((3, 0, 1, 2))

    return v


class Echo(torchvision.datasets.VisionDataset):
    """EchoNet-Dynamic Dataset.

    Args:
        root (string): Root directory of dataset (defaults to `echonet.config.DATA_DIR`)
        split (string): One of {``train'', ``val'', ``test'', ``all'', or ``external_test''}
        target_type (string or list, optional): Type of target to use,
            ``Filename'', ``EF'', ``EDV'', ``ESV'', ``LargeIndex'',
            ``SmallIndex'', ``LargeFrame'', ``SmallFrame'', ``LargeTrace'',
            or ``SmallTrace''
            Can also be a list to output a tuple with all specified target types.
            The targets represent:
                ``Filename'' (string): filename of video
                ``EF'' (float): ejection fraction
                ``EDV'' (float): end-diastolic volume
                ``ESV'' (float): end-systolic volume
                ``LargeIndex'' (int): index of large (diastolic) frame in video
                ``SmallIndex'' (int): index of small (systolic) frame in video
                ``LargeFrame'' (np.array shape=(3, height, width)): normalized large (diastolic) frame
                ``SmallFrame'' (np.array shape=(3, height, width)): normalized small (systolic) frame
                ``LargeTrace'' (np.array shape=(height, width)): left ventricle large (diastolic) segmentation
                    value of 0 indicates pixel is outside left ventricle
                             1 indicates pixel is inside left ventricle
                ``SmallTrace'' (np.array shape=(height, width)): left ventricle small (systolic) segmentation
                    value of 0 indicates pixel is outside left ventricle
                             1 indicates pixel is inside left ventricle
            Defaults to ``EF''.
        mean (int, float, or np.array shape=(3,), optional): means for all (if scalar) or each (if np.array) channel.
            Used for normalizing the video. Defaults to 0 (video is not shifted).
        std (int, float, or np.array shape=(3,), optional): standard deviation for all (if scalar) or each (if np.array) channel.
            Used for normalizing the video. Defaults to 0 (video is not scaled).
        length (int or None, optional): Number of frames to clip from video. If ``None'', longest possible clip is returned.
            Defaults to 16.
        period (int, optional): Sampling period for taking a clip from the video (i.e. every ``period''-th frame is taken)
            Defaults to 2.
        max_length (int or None, optional): Maximum number of frames to clip from video (main use is for shortening excessively
            long videos when ``length'' is set to None). If ``None'', shortening is not applied to any video.
            Defaults to 250.
        clips (int, optional): Number of clips to sample. Main use is for test-time augmentation with random clips.
            Defaults to 1.
        pad (int or None, optional): Number of pixels to pad all frames on each side (used as augmentation).
            and a window of the original size is taken. If ``None'', no padding occurs.
            Defaults to ``None''.
        noise (float or None, optional): Fraction of pixels to black out as simulated noise. If ``None'', no simulated noise is added.
            Defaults to ``None''.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        external_test_location (string): Path to videos to use for external testing.
    """

    def __init__(
        self,
        root,
        split="train",
        target_type="EF",
        mean=MEAN,
        std=STD,
        length=16,
        period=2,
        max_length=250,
        clips=1,
        pad=None,
        noise=None,
        target_transform=None,
        external_test_location=None,
    ):

        super().__init__(root, target_transform=target_transform)

        self.split = split.upper()
        if not isinstance(target_type, list):
            target_type = [target_type]
        self.target_type = target_type
        self.mean = mean
        self.std = std
        self.length = length
        self.max_length = max_length
        self.period = period
        self.clips = clips
        self.pad = pad
        self.noise = noise
        self.target_transform = target_transform
        self.external_test_location = external_test_location

        self.fnames, self.outcome = [], []

        if self.split == "EXTERNAL_TEST":
            self.fnames = sorted(os.listdir(self.external_test_location))
        else:
            # Load video-level labels
            with open(os.path.join(self.root, "FileList.csv")) as f:
                data = pandas.read_csv(f)
            data["Split"].map(lambda x: x.upper())

            if self.split != "ALL":
                data = data[data["Split"] == self.split]

            self.header = data.columns.tolist()
            self.fnames = data["FileName"].tolist()
            self.fnames = [
                fn + ".avi" for fn in self.fnames if os.path.splitext(fn)[1] == ""
            ]  # Assume avi if no suffix
            self.outcome = data.values.tolist()

            # Check that files are present
            missing = set(self.fnames) - set(
                os.listdir(os.path.join(self.root, "Videos"))
            )
            if len(missing) != 0:
                print(
                    "{} videos could not be found in {}:".format(
                        len(missing), os.path.join(self.root, "Videos")
                    )
                )
                for f in sorted(missing):
                    print("\t", f)
                raise FileNotFoundError(
                    os.path.join(self.root, "Videos", sorted(missing)[0])
                )

            # Load traces
            self.frames = collections.defaultdict(list)
            self.trace = collections.defaultdict(_defaultdict_of_lists)

            with open(os.path.join(self.root, "VolumeTracings.csv")) as f:
                header = f.readline().strip().split(",")
                assert header == ["FileName", "X1", "Y1", "X2", "Y2", "Frame"]

                for line in f:
                    filename, x1, y1, x2, y2, frame = line.strip().split(",")
                    x1 = float(x1)
                    y1 = float(y1)
                    x2 = float(x2)
                    y2 = float(y2)
                    frame = int(frame)
                    if frame not in self.trace[filename]:
                        self.frames[filename].append(frame)
                    self.trace[filename][frame].append((x1, y1, x2, y2))
            for filename in self.frames:
                for frame in self.frames[filename]:
                    self.trace[filename][frame] = np.array(self.trace[filename][frame])

            # A small number of videos are missing traces; remove these videos
            keep = [len(self.frames[f]) >= 2 for f in self.fnames]
            self.fnames = [f for (f, k) in zip(self.fnames, keep) if k]
            self.outcome = [f for (f, k) in zip(self.outcome, keep) if k]

    def __getitem__(self, index):
        # Find filename of video
        if self.split == "EXTERNAL_TEST":
            video = os.path.join(self.external_test_location, self.fnames[index])
        elif self.split == "CLINICAL_TEST":
            video = os.path.join(
                self.root, "ProcessedStrainStudyA4c", self.fnames[index]
            )
        else:
            video = os.path.join(self.root, "Videos", self.fnames[index])

        # Load video into np.array
        video = loadvideo(video).astype(float)

        # Add simulated noise (black out random pixels)
        # 0 represents black at this point (video has not been normalized yet)
        if self.noise is not None:
            n = video.shape[1] * video.shape[2] * video.shape[3]
            ind = np.random.choice(n, round(self.noise * n), replace=False)
            f = ind % video.shape[1]
            ind //= video.shape[1]
            i = ind % video.shape[2]
            ind //= video.shape[2]
            j = ind
            video[:, f, i, j] = 0

        # Apply normalization
        if isinstance(self.mean, (float, int)):
            video -= self.mean
        else:
            video -= self.mean.reshape(3, 1, 1, 1)

        if isinstance(self.std, (float, int)):
            video /= self.std
        else:
            video /= self.std.reshape(3, 1, 1, 1)

        # Set number of frames
        c, f, h, w = video.shape
        if self.length is None:
            # Take as many frames as possible
            length = f // self.period
        else:
            # Take specified number of frames
            length = self.length

        if self.max_length is not None:
            # Shorten videos to max_length
            length = min(length, self.max_length)

        if f < length * self.period:
            # Pad video with frames filled with zeros if too short
            # 0 represents the mean color (dark grey), since this is after normalization
            video = np.concatenate(
                (video, np.zeros((c, length * self.period - f, h, w), video.dtype)),
                axis=1,
            )
            c, f, h, w = video.shape  # pylint: disable=E0633

        if self.clips == "all":
            # Take all possible clips of desired length
            start = np.arange(f - (length - 1) * self.period)
        else:
            # Take random clips from video
            start = np.random.choice(f - (length - 1) * self.period, self.clips)

        # Gather targets
        target = []
        for t in self.target_type:
            key = self.fnames[index]
            if t == "Filename":
                target.append(self.fnames[index])
            elif t == "LargeIndex":
                # Traces are sorted by cross-sectional area
                # Largest (diastolic) frame is last
                target.append(np.int(self.frames[key][-1]))
            elif t == "SmallIndex":
                # Largest (diastolic) frame is first
                target.append(np.int(self.frames[key][0]))
            elif t == "LargeFrame":
                target.append(video[:, self.frames[key][-1], :, :])
            elif t == "SmallFrame":
                target.append(video[:, self.frames[key][0], :, :])
            elif t in ["LargeTrace", "SmallTrace"]:
                if t == "LargeTrace":
                    t = self.trace[key][self.frames[key][-1]]
                else:
                    t = self.trace[key][self.frames[key][0]]
                x1, y1, x2, y2 = t[:, 0], t[:, 1], t[:, 2], t[:, 3]
                x = np.concatenate((x1[1:], np.flip(x2[1:])))
                y = np.concatenate((y1[1:], np.flip(y2[1:])))

                r, c = skimage.draw.polygon(
                    np.rint(y).astype(int),
                    np.rint(x).astype(int),
                    (video.shape[2], video.shape[3]),
                )
                mask = np.zeros((video.shape[2], video.shape[3]), np.float32)
                mask[r, c] = 1
                target.append(mask)
            else:
                if self.split == "CLINICAL_TEST" or self.split == "EXTERNAL_TEST":
                    target.append(np.float32(0))
                else:
                    target.append(np.float32(self.outcome[index][self.header.index(t)]))

        if target != []:
            target = tuple(target) if len(target) > 1 else target[0]
            if self.target_transform is not None:
                target = self.target_transform(target)

        # Select clips from video
        video = tuple(
            video[:, s + self.period * np.arange(length), :, :] for s in start
        )
        if self.clips == 1:
            video = video[0]
        else:
            video = np.stack(video)

        if self.pad is not None:
            # Add padding of zeros (mean color of videos)
            # Crop of original size is taken out
            # (Used as augmentation)
            c, l, h, w = video.shape
            temp = np.zeros(
                (c, l, h + 2 * self.pad, w + 2 * self.pad), dtype=video.dtype
            )
            temp[:, :, self.pad : -self.pad, self.pad : -self.pad] = (
                video  # pylint: disable=E1130
            )
            i, j = np.random.randint(0, 2 * self.pad, 2)
            video = temp[:, :, i : (i + h), j : (j + w)]

        video = rearrange(video[0], "F H W -> F () H W")
        video = torch.tensor(video, dtype=torch.float32)
        return {"image": video, "target": target}

    def __len__(self):
        return len(self.fnames)

    def extra_repr(self) -> str:
        """Additional information to add at end of __repr__."""
        lines = ["Target type: {target_type}", "Split: {split}"]
        return "\n".join(lines).format(**self.__dict__)


def _defaultdict_of_lists():
    """Returns a defaultdict of lists.

    This is used to avoid issues with Windows (if this function is anonymous,
    the Echo dataset cannot be used in a dataloader).
    """

    return collections.defaultdict(list)


class EchoNetDataModuleV2(L.LightningDataModule):
    """
    DataModule for the EchoNet dataset.
    This class is used to load the data, create the dataloaders and handle the specific taks transformations.
    Can be regression of output_variable or image generation with channel or pixel wise dropout.
    """

    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 12,
        length=16,
        period=2,
        max_length=250,
        clips=1,
        pad=None,
        noise=None,
        target_transform=None,
    ):
        """
        Initializes the Echonet dataset.

        Args:
            batch_size (int): The batch size for data loading. Default is 32.
            num_workers (int): The number of worker threads for data loading. Default is 12.
            length (int): The length of video clips. Default is 16.
            period (int): The sampling rate for the video. Default is 2.
            max_length (int): The maximum length of video clips. Default is 250.
            clips (int): The number of clips to sample. Default is 1.
            pad (int): The number of pixels to pad all frames on each side. Default is None.
            noise (float): The fraction of pixels to black out as simulated noise. Default is None.
            target_transform (callable): A function/transform that takes in the target and transforms it. Default is None.
        """
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.length = length
        self.period = period
        self.max_length = max_length
        self.clips = clips
        self.pad = pad
        self.noise = noise
        self.target_transform = target_transform

        self.save_hyperparameters()

    def setup(self, stage: str = None):
        self.train = Echo(
            root=ROOT,
            split="train",
            length=self.length,
            period=self.period,
            max_length=self.max_length,
            clips=self.clips,
            pad=self.pad,
            noise=self.noise,
            target_transform=self.target_transform,
        )
        self.val = Echo(
            root=ROOT,
            split="val",
            length=self.length,
            period=self.period,
            max_length=self.max_length,
            clips=self.clips,
            pad=self.pad,
            noise=self.noise,
            target_transform=self.target_transform,
        )
        self.test = Echo(
            root=ROOT,
            split="test",
            length=self.length,
            period=self.period,
            max_length=self.max_length,
            clips=self.clips,
            pad=self.pad,
            noise=self.noise,
            target_transform=self.target_transform,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test, batch_size=self.batch_size, num_workers=self.num_workers
        )
