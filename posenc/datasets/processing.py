from typing import List, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def extract_data_summary(
    dataset: torch.utils.data.Dataset,
    foreground_threshold: Optional[float] = None,
    image_key: Optional[str] = None,
    num_workers: int = 0,
    interval: int = 100,
    min_percentile: float = 0.5,
    max_percentile: float = 99.5,
    clip_percentiles_upper: Optional[list] = None,
    clip_percentiles_lower: Optional[list] = None,
):
    """
    Extracts summary statistics from a dataset.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to extract summary statistics from.
        foreground_threshold (float, optional): Threshold value for foreground pixels. If specified,
            only pixels above this threshold will be considered. Defaults to None.
        image_key (str, optional): Key to access the image data in the batch. If None, assumes the
            image is the first element in the batch. Defaults to None.
        num_workers (int, optional): Number of worker processes to use for data loading. Defaults to 0.
        interval (int, optional): Interval for sampling intensities. Defaults to 10.
        min_percentile (float, optional): Percentile value for minimum intensities. Defaults to 0.5.
        max_percentile (float, optional): Percentile value for maximum intensities. Defaults to 99.5.

    Returns:
        dict: A dictionary containing the following summary statistics:
            - "mean": Mean intensity per channel.
            - "std": Standard deviation of intensity per channel.
            - "min": Minimum intensity per channel.
            - "max": Maximum intensity per channel.
            - "median": Median intensity per channel.
            - "min_percentiles_{min_percentile}": Minimum percentile intensity per channel.
            - "max_percentiles_{max_percentile}": Maximum percentile intensity per channel.
            - "voxel_count": Total count of voxels.
            - "mean_image": Mean image computed from the dataset.
            - "all_intensities": List of intensities sampled from the dataset.
    """

    def _get_image(batch) -> torch.Tensor:
        """Get image and remove batch dimension."""
        if image_key is None:
            img, _ = batch
            return img[0]
        return batch[image_key][0]

    def _process_channel(
        channel: torch.Tensor,
        voxel_sum: torch.Tensor,
        voxel_square_sum: torch.Tensor,
        voxel_min: List,
        voxel_max: List,
        all_intensities: List,
    ):
        """Process a single channel of the image."""
        voxel_sum += channel.sum()
        voxel_square_sum += torch.square(channel).sum()
        voxel_min.append(channel.min().item())
        voxel_max.append(channel.max().item())
        all_intensities += channel.flatten()[::interval].tolist()
        return len(channel)

    # Create a DataLoader to iterate over the dataset
    loader = DataLoader(dataset, num_workers=num_workers)

    # Extract an example image to get the shape of the image
    example_img = _get_image(next(iter(loader)))
    mean_image = torch.zeros_like(example_img)
    n_channels = example_img.shape[0]

    # Initialize variables to store the summary statistics
    all_intensities = [[] for _ in range(n_channels)]
    voxel_sum = torch.zeros(n_channels)
    voxel_square_sum = torch.zeros(n_channels)
    voxel_min = [[] for _ in range(n_channels)]
    voxel_max = [[] for _ in range(n_channels)]
    voxel_count = 0

    for batch in tqdm(loader, total=len(loader)):
        img = _get_image(batch)
        mean_image += img / len(dataset)

        for c in range(n_channels):

            if foreground_threshold is not None:
                channel = img[c, img[c, ...] > foreground_threshold]
            else:
                channel = img[c]

            if (
                clip_percentiles_lower is not None
                and clip_percentiles_upper is not None
            ):
                channel = torch.clip(
                    channel,
                    clip_percentiles_lower[c],
                    clip_percentiles_upper[c],
                )

            voxel_count += _process_channel(
                channel,
                voxel_sum[c],
                voxel_square_sum[c],
                voxel_min[c],
                voxel_max[c],
                all_intensities[c],
            )

    voxel_count_per_channel = voxel_count / n_channels

    mean = voxel_sum / voxel_count_per_channel
    std = torch.sqrt(voxel_square_sum / voxel_count_per_channel - mean**2)
    data_min = [min(voxel_min[i]) for i in range(n_channels)]
    data_max = [max(voxel_max[i]) for i in range(n_channels)]
    median = [
        torch.median(torch.tensor(all_intensities[i])).item() for i in range(n_channels)
    ]
    min_percentiles = [
        torch.quantile(torch.tensor(all_intensities[i]), q=min_percentile / 100).item()
        for i in range(n_channels)
    ]
    max_percentiles = [
        torch.quantile(torch.tensor(all_intensities[i]), q=max_percentile / 100).item()
        for i in range(n_channels)
    ]

    return {
        "mean": mean,
        "std": std,
        "min": data_min,
        "max": data_max,
        "median": median,
        f"min_percentiles_{min_percentile}": min_percentiles,
        f"max_percentiles_{max_percentile}": max_percentiles,
        "voxel_count": voxel_count,
        "mean_image": mean_image,
        "all_intensities": all_intensities,
    }
