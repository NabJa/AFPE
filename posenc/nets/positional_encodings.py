from typing import List, Union

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
from monai.networks.layers import Conv, trunc_normal_
from monai.utils import ensure_tuple_rep

from posenc.enums import PatchEmbeddingType, PosEncType


def process_grid_size(grid_size: Union[int, List[int]], spatial_dims: int) -> List[int]:
    """
    Process the grid size based on the spatial dimensions.

    Args:
        grid_size: The size of the grid. It can be either an integer or a list of integers.
        spatial_dims: The number of spatial dimensions.

    Returns:
        List[int]: The processed grid size.

    """
    if isinstance(grid_size, list):
        return grid_size
    return [grid_size for _ in range(spatial_dims)]

def isotropic_fpe(
    grid_size,
    hidden_size: int,
    spatial_dims: int,
    scale: float = 1.0,
    variance_factors: torch.Tensor = None,
    n_tokens: int = 0,
) -> torch.Tensor:
    """
    Anisotropic Fourier Feature Positional Encodings (AFPE).
    
    Generates Fourier feature positional encodings with customizable variance in specified dimensions.

    Args:
        grid_size:          The size of the grid in each spatial dimension.
        hidden_size:        The size of the hidden dimension.
        spatial_dims:       The number of spatial dimensions.
        scale:              The scaling factor for the distribution. Defaults to 1.0.
        variance_factors:   A tensor specifying variance factors (or scale hyperparameters s_i)
                            for each spatial dimension. If None, all variances are 1.
        n_tokens:           Number of additional token encodings. Defaults to 0.

    Returns:
        torch.Tensor: The positional encodings.

    Raises:
        AssertionError: If the hidden_size is not divisible by 2.
        ValueError: If variance_factors is not the same length as spatial_dims.
    """
    assert hidden_size % 2 == 0

    grid_size = process_grid_size(grid_size, spatial_dims)

    # Set variance factors to 1 if not provided
    if variance_factors is None:
        variance_factors = torch.tensor(grid_size) / max(grid_size)
    else:
        variance_factors = torch.tensor(variance_factors, dtype=torch.float32)

    # Ensure variance_factors matches the number of spatial dimensions
    if len(variance_factors) != spatial_dims:
        raise ValueError(
            "Length of variance_factors must match the number of spatial dimensions."
        )

    # Adjust scale of each dimension by the variance factor
    adjusted_scale = scale * variance_factors

    # The distribution is sampled from a normal distribution with mean 0 and standard deviation adjusted_scale.
    distribution = (
        torch.normal(0.0, 1.0, (hidden_size // 2, spatial_dims)) * adjusted_scale
    )

    # Positions are mapped between 0 and 1
    positions = [torch.linspace(0, 1, x) for x in grid_size]

    # All positions are stacked on a meshgrid.
    positions = torch.stack(torch.meshgrid(*positions, indexing="ij"), axis=-1)
    positions = rearrange(positions, "... d -> (...) d")

    # The positions are then multiplied with the distribution and summed up to get the final positional encoding.
    x_proj = (2.0 * torch.pi * positions) @ distribution.T

    # The final positional encoding is then concatenated with the sin and cos of the projected positions.
    pe = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], axis=-1)

    if n_tokens > 0:
        token_positions = [torch.linspace(2, 3, n_tokens) for x in range(spatial_dims)]
        token_positions = torch.stack(
            torch.meshgrid(*token_positions, indexing="ij"), axis=-1
        )
        token_positions = token_positions[0, ...]
        token_proj = (2.0 * torch.pi * token_positions) @ distribution.T
        token_pe = torch.cat([torch.sin(token_proj), torch.cos(token_proj)], axis=-1)
        pe = torch.cat([token_pe, pe])

    return rearrange(pe, "n d -> 1 n d")


def sincos_encode_single_position(p, dim=512, temp=10_000):
    pe = torch.FloatTensor([p / (temp ** (2 * (i // 2) / dim)) for i in range(dim)])
    pe[0::2] = torch.sin(pe[0::2])
    pe[1::2] = torch.cos(pe[1::2])
    return rearrange(pe, "... -> 1 ...")


def sincos_position_encoding(
    grid_size: Union[int, List[int]],
    hidden_size: int,
    spatial_dims: int,
    temperature: float = 10000.0,
    num_tokens: int = 0,
) -> torch.Tensor:
    """
    Generate sinusoidal positional encodings.

    Args:
        grid_size: The size of the grid. It can be either an integer or a list of integers.
        hidden_size: The size of the hidden dimension.
        spatial_dims: The number of spatial dimensions.
        temperature: The temperature parameter for the positional encoding.
        num_tokens: The number of tokens to encode. Will be added to the beginnin! Defaults to 0.

    Returns:
        torch.Tensor: The positional encodings.

    """
    grid_size = process_grid_size(grid_size, spatial_dims)
    unseen_position = max(grid_size) + 1

    positions = [torch.arange(x, dtype=torch.float32) for x in grid_size]
    positions = torch.meshgrid(*positions, indexing="ij")

    pos_dim = hidden_size // (spatial_dims * 2)

    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1.0 / (temperature**omega)

    out = [torch.einsum("m,d->md", [grid.flatten(), omega]) for grid in positions]
    out = torch.cat([torch.sin(o) for o in out] + [torch.cos(o) for o in out], dim=-1)

    if num_tokens > 0:
        token_encodings = [
            sincos_encode_single_position(p, dim=hidden_size, temp=temperature)
            for p in range(unseen_position, unseen_position + num_tokens)
        ]
        out = torch.cat([*token_encodings, out])

    return out


def fourier_feature_position_encoding(
    grid_size,
    hidden_size: int,
    spatial_dims: int,
    scale: float = 1.0,
    n_tokens: int = 0,
) -> torch.Tensor:
    """
    Generates Fourier feature positional encodings.

    Args:
        grid_size: The size of the grid in each spatial dimension.
        hidden_size: The size of the hidden dimension.
        spatial_dims: The number of spatial dimensions.
        scale: The scaling factor for the distribution. Defaults to 1.0.

    Returns:
        torch.Tensor: The positional encodings.

    Raises:
        AssertionError: If the hidden_size is not divisible by 2.
    """
    assert hidden_size % 2 == 0

    grid_size = process_grid_size(grid_size, spatial_dims)

    # The distribution is sampled from a normal distribution with mean 0 and standard deviation 1.
    distribution = torch.normal(0.0, 1.0, (hidden_size // 2, spatial_dims)) * scale

    # Positions are mapped between 0 and 1
    positions = [torch.linspace(0, 1, x) for x in grid_size]

    # All positions are stacked on a meshgrid.
    # E.g. grid=[16, 16] will result in positions=[16, 16, 2] where are the x and y coordinates are normalized between 0 and 1
    positions = torch.stack(torch.meshgrid(*positions, indexing="ij"), axis=-1)
    positions = rearrange(positions, "... d -> (...) d")

    # The positions are then multiplied with the distribution and summed up to get the final positional encoding.
    x_proj = (2.0 * torch.pi * positions) @ distribution.T

    # The final positional encoding is then concatenated with the sin and cos of the projected positions.
    pe = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], axis=-1)

    if n_tokens > 0:
        token_positions = [torch.linspace(2, 3, n_tokens) for x in range(spatial_dims)]
        token_positions = torch.stack(
            torch.meshgrid(*token_positions, indexing="ij"), axis=-1
        )
        token_positions = token_positions[0, ...]
        token_proj = (2.0 * torch.pi * token_positions) @ distribution.T
        token_pe = torch.cat([torch.sin(token_proj), torch.cos(token_proj)], axis=-1)
        pe = torch.cat([token_pe, pe])

    return rearrange(pe, "n d -> 1 n d")




class LearnableFPE(nn.Module):
    """Learnable Fourier Positional Encoding module."""

    def __init__(self, grid_size, hidden_size, spatial_dims, n_tokens=0):
        """
        Args:
            grid_size: The size of the grid in each spatial dimension.
            hidden_size: The size of the hidden dimension.
            spatial_dims: The number of spatial dimensions.
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.n_tokens = n_tokens

        self.pos_emb = torch.normal(0.0, 1.0, (hidden_size // 2, spatial_dims))
        self.pos_emb = nn.Parameter(self.pos_emb)

        self.grid_size = process_grid_size(grid_size, spatial_dims)
        self.n_patches = torch.prod(torch.tensor(self.grid_size))
        self.positions = [torch.linspace(0, 1, x) for x in self.grid_size]
        self.positions = torch.stack(torch.meshgrid(*self.positions), dim=-1)

        if n_tokens > 0:
            self.token_positions = [
                torch.linspace(2, 3, n_tokens) for x in range(spatial_dims)
            ]
            self.token_positions = torch.stack(
                torch.meshgrid(*self.token_positions, indexing="ij"), axis=-1
            )
            self.token_positions = self.token_positions[0, ...]

    def forward(self):
        """
        The forward pass uses the positional embeddings to generate
        the final positional encodings by using the parameters (pos_emb) and
        projecting them into fourier features.
        """
        x_proj = (
            2.0 * torch.pi * self.positions.to(self.pos_emb.device)
        ) @ self.pos_emb.T
        pe = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], axis=-1)
        pe = rearrange(pe, "... d -> (...) d")

        if self.n_tokens > 0:
            token_proj = (
                2.0 * torch.pi * self.token_positions.to(self.pos_emb.device)
            ) @ self.pos_emb.T
            token_pe = torch.cat(
                [torch.sin(token_proj), torch.cos(token_proj)], axis=-1
            )
            pe = torch.cat([token_pe, pe])

        return rearrange(pe, "n d -> 1 n d")


class LearnableScaleFPE(nn.Module):
    """Fourier Positional Encoding where the scale are learnable parameters module."""

    def __init__(self, grid_size, hidden_size, spatial_dims):
        """
        Args:
            grid_size: The size of the grid in each spatial dimension.
            hidden_size: The size of the hidden dimension.
            spatial_dims: The number of spatial dimensions.
        """
        super().__init__()

        self.hidden_size = hidden_size

        self.pos_emb = torch.normal(0.0, 1.0, (hidden_size // 2, spatial_dims))
        self.scale = nn.Parameter(torch.ones(spatial_dims))

        self.grid_size = process_grid_size(grid_size, spatial_dims)
        self.n_patches = torch.prod(torch.tensor(self.grid_size))
        self.positions = [torch.linspace(0, 1, x) for x in self.grid_size]
        self.positions = torch.stack(torch.meshgrid(*self.positions), dim=-1)

    def forward(self):
        """
        The forward pass uses the positional embeddings to generate
        the final positional encodings by using the parameters (pos_emb) and
        projecting them into fourier features.
        """
        pos_emb = self.pos_emb * self.scale.to(self.pos_emb.device)
        x_proj = (2.0 * torch.pi * self.positions.to(pos_emb.device)) @ pos_emb.T
        pe = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], axis=-1)
        return rearrange(pe, "... d -> 1 (...) d")


class PositionalEmbedding(nn.Module):
    """Positional embedding module for use in Vision Transformer models."""

    def __init__(
        self,
        pos_embed_type: Union[str, PosEncType],
        img_size: Union[int, List[int]],
        patch_size: Union[int, List[int]],
        hidden_size: int,
        spatial_dims: int,
        temperature: float = 10e3,
        scale: float = 1,
        addition_type: str = "add",
        n_tokens: int = 0,
        variance_factors: torch.Tensor = None,
    ):
        """
        Args:
            pos_embed_type: The type of positional embedding to use.
            img_size: The size of the input image.
            patch_size: The size of the patches.
            hidden_size: The size of the hidden dimension.
            spatial_dims: The number of spatial dimensions.
            temperature: The temperature parameter for positional encodings. Defaults to 10e3.
            scale: The scaling factor for Fourier feature positional encodings. Defaults to 1.
            addition_type: The type of addition to use for combining positional encodings with input. Defaults to "add".
            n_tokens: The number of tokens to encode. Defaults to 0.
            variance_factors: The variance factors for isotropic positional encodings. Defaults to None.
        """
        super().__init__()

        self.img_size = ensure_tuple_rep(img_size, spatial_dims)
        self.patch_size = ensure_tuple_rep(patch_size, spatial_dims)
        self.hidden_size = hidden_size
        self.spatial_dims = spatial_dims
        self.addition_type = addition_type
        self.n_tokens = n_tokens
        self.temperature = temperature
        self.scale = scale
        self.pos_embed_type = (
            PosEncType(pos_embed_type)
            if isinstance(pos_embed_type, str)
            else pos_embed_type
        )
        self.variance_factors = variance_factors

        self.grid_size = [i // p for i, p in zip(self.img_size, self.patch_size)]
        self.n_patches = np.prod(self.grid_size)

        # Make argument checks
        self._check_hidden_dims()
        self._check_patch_size()

        # Build positional encodings
        self.positions = self._build_pos_enc(self.pos_embed_type)

    def _check_hidden_dims(self) -> None:
        """Check if the hidden size is valid."""
        if self.spatial_dims == 2:
            assert (
                self.hidden_size % 2 == 0
            ), "In 2D the hidden_size must be divisable by 2."
        elif self.spatial_dims == 3:
            assert (
                self.hidden_size % 6 == 0
            ), "In 3D the hidden_size must be divisable by 6."
        else:
            raise NotImplementedError("Only 2D and 3D is supported.")

    def _check_patch_size(self) -> None:
        """Check if the patch size is valid."""
        for m, p in zip(self.img_size, self.patch_size):
            if m < p:
                raise ValueError("patch_size should be smaller than img_size.")

    def _build_pos_enc(self, pos_embed_type: PosEncType) -> nn.Parameter:
        """Builds positional encodings based on the specified type."""

        if pos_embed_type == PosEncType.LFPE:
            return LearnableFPE(
                self.grid_size,
                self.hidden_size,
                self.spatial_dims,
                n_tokens=self.n_tokens,
            )
        if pos_embed_type == PosEncType.LSFPE:
            return LearnableScaleFPE(
                self.grid_size, self.hidden_size, self.spatial_dims
            )

        pos_emb = nn.Parameter(
            torch.zeros(1, self.n_patches + self.n_tokens, self.hidden_size),
            requires_grad=pos_embed_type == PosEncType.LEARNABLE,
        )

        if pos_embed_type == PosEncType.LEARNABLE:
            pos_emb = trunc_normal_(pos_emb, mean=0.0, std=0.02, a=-2.0, b=2.0)

        with torch.no_grad():
            if pos_embed_type == PosEncType.SINCOS:
                pos_embeddings = sincos_position_encoding(
                    self.grid_size,
                    self.hidden_size,
                    self.spatial_dims,
                    temperature=self.temperature,
                    num_tokens=self.n_tokens,
                )
                pos_emb.data.copy_(pos_embeddings.float())

            if pos_embed_type == PosEncType.FOURIER:
                pos_embeddings = fourier_feature_position_encoding(
                    self.grid_size,
                    self.hidden_size,
                    self.spatial_dims,
                    scale=self.scale,
                    n_tokens=self.n_tokens,
                )
                pos_emb.data.copy_(pos_embeddings)

            if pos_embed_type == PosEncType.ISOFPE:
                pos_embeddings = isotropic_fpe(
                    self.grid_size,
                    self.hidden_size,
                    self.spatial_dims,
                    scale=self.scale,
                    n_tokens=self.n_tokens,
                    variance_factors=self.variance_factors,
                )
                pos_emb.data.copy_(pos_embeddings)

        if pos_embed_type != PosEncType.LEARNABLE:
            pos_emb.requires_grad = False

        return pos_emb

    def forward(self, x):
        if (self.pos_embed_type == PosEncType.LFPE) or (
            self.pos_embed_type == PosEncType.LSFPE
        ):
            # LearnableFPE have to be projected every time forward is called
            p = self.positions()
        else:
            p = self.positions

        if self.addition_type == "add":
            return x + p.to(x.device)

        return torch.concatenate(x, p)


class VideoPatchEmbedding(nn.Module):
    def __init__(self, patch_size, hidden_size):
        super().__init__()

        if isinstance(patch_size, list):
            num_voxels = np.prod(patch_size)
            patch_size = patch_size[1]
        else:
            num_voxels = patch_size**3

        self.encode = nn.Sequential(
            Rearrange(
                "b f () (h p1) (w p2) -> b (f h w) (p1 p2)",
                p1=patch_size,
                p2=patch_size,
            ),
            nn.LayerNorm(num_voxels),
            nn.Linear(num_voxels, hidden_size),
            nn.LayerNorm(hidden_size),
        )

    def forward(self, x):
        return self.encode(x)


class VideoPatchEmbeddingV2(nn.Module):
    def __init__(self, patch_size, hidden_size, conv=True):
        super().__init__()

        self.conv = conv

        patch_f, patch_h, patch_w = patch_size
        num_voxels = np.prod(patch_size)

        if conv:
            self.encode = nn.Sequential(
                Rearrange("b f c h w -> b c f h w"),
                nn.Conv3d(1, hidden_size, kernel_size=patch_size, stride=patch_size),
                Rearrange("b c f h w -> b (f h w) c"),
                nn.LayerNorm(hidden_size),
            )
        else:
            self.encode = nn.Sequential(
                Rearrange(
                    "b (f p0) () (h p1) (w p2) -> b (f h w) (p0 p1 p2)",
                    p0=patch_f,
                    p1=patch_h,
                    p2=patch_w,
                ),
                nn.LayerNorm(num_voxels),
                nn.Linear(num_voxels, hidden_size),
                nn.LayerNorm(hidden_size),
            )

    def forward(self, x):
        return self.encode(x)


class ConvPatchEmbedding(nn.Module):
    def __init__(self, spatial_dims, patch_size=16, hidden_size=768, in_channels=1):
        super().__init__()

        self.encode = nn.Sequential(
            Conv[Conv.CONV, spatial_dims](
                in_channels=in_channels,
                out_channels=hidden_size,
                kernel_size=patch_size,
                stride=patch_size,
            ),
            Rearrange(
                "B C ... -> B (...) C"
            ),  # Batch Channels *Spatial_dims to Batch spatial_dims_flattened Channels
        )

    def forward(self, x):
        return self.encode(x)


class PatchEmbeddingBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        img_size: Union[int, List[int]],
        patch_size: Union[int, List[int]],
        hidden_size: int,
        num_heads: int,
        pos_embed_type: Union[str, PosEncType] = PosEncType.SINCOS,
        patch_embed_type: PatchEmbeddingType = PatchEmbeddingType.CONV,
        dropout_rate: float = 0.0,
        spatial_dims: int = 2,
        scale: float = 1,
        temperature: float = 10000,
        variance_factors: torch.Tensor = None,
    ) -> None:
        """
        Args:
            in_channels (int): Number of input channels.
            img_size (Union[int, List[int]]): Size of the input image.
            patch_size (Union[int, List[int]]): Size of each patch.
            hidden_size (int): Size of the hidden dimension.
            num_heads (int): Number of attention heads.
            pos_embed_type (str, optional): Type of positional embedding. Defaults to "sincos".
            dropout_rate (float, optional): Dropout rate. Defaults to 0.0.
            spatial_dims (int, optional): Number of spatial dimensions. Defaults to 2.
            scale (float, optional): Scale factor for positional encoding. Defaults to 1.
            temperature (float, optional): Temperature for positional encoding. Defaults to 10000.
            variance_factors (torch.Tensor, optional): Variance factors for isotropic positional encodings. Defaults to None.
        """
        super().__init__()

        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden size {hidden_size} should be divisible by num_heads {num_heads}."
            )

        self.spatial_dims = spatial_dims
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.in_channels = in_channels

        # Build the patch embeddings module
        if not isinstance(patch_embed_type, PatchEmbeddingType):
            patch_embed_type = PatchEmbeddingType(patch_embed_type)

        self.patch_embeddings = self._build_patch_embeddings(patch_embed_type)

        # Build the positional embeddings module
        if not isinstance(pos_embed_type, PosEncType):
            pos_embed_type = PosEncType(pos_embed_type)

        self.position_embeddings = PositionalEmbedding(
            pos_embed_type,
            img_size,
            patch_size,
            hidden_size,
            spatial_dims,
            temperature,
            scale,
            variance_factors=variance_factors,
        )

        # Dropout optional
        self.dropout = nn.Dropout(dropout_rate)

    def _build_patch_embeddings(self, patch_embed_type: PatchEmbeddingType):
        if patch_embed_type == PatchEmbeddingType.CONV:
            return ConvPatchEmbedding(
                self.spatial_dims,
                self.patch_size,
                self.hidden_size,
                self.in_channels,
            )
        elif patch_embed_type == PatchEmbeddingType.VIDEO:
            return VideoPatchEmbedding(self.patch_size, self.hidden_size)
        elif patch_embed_type == PatchEmbeddingType.VIDEOV2:
            return VideoPatchEmbeddingV2(self.patch_size, self.hidden_size, conv=True)
        else:
            raise ValueError(f"Unsupported patch embedding type: {patch_embed_type}")

    def forward(self, x):
        # Run convolution over every patch
        x = self.patch_embeddings(x)

        # Add positional encoding of every patch
        x = self.position_embeddings(x)

        x = self.dropout(x)

        return x
