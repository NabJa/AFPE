from functools import partial
from typing import Sequence

import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from monai.networks.blocks.patchembedding import (
    PatchEmbeddingBlock as PatchEmbeddingBlockMonai,
)
from torch import nn

from posenc.enums import PatchEmbeddingType, PosEncType
from posenc.nets.blocks import Transformer, TransformerBlock
from posenc.nets.positional_encodings import PatchEmbeddingBlock, PositionalEmbedding


def pair(t, n=2):
    return t if isinstance(t, tuple) else [t for _ in range(n)]

class ViTMonai(nn.Module):
    """
    Vision Transformer (ViT), based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"

    ViT supports Torchscript but only works for Pytorch after 1.8.
    """

    def __init__(
        self,
        in_channels: int,
        img_size: Sequence[int] | int,
        patch_size: Sequence[int] | int,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        proj_type: str = "conv",
        pos_embed_type: str = "learnable",
        classification: bool = False,
        num_classes: int = 2,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        post_activation="Tanh",
        qkv_bias: bool = False,
        save_attn: bool = False,
    ) -> None:
        """
        Args:
            in_channels (int): dimension of input channels.
            img_size (Union[Sequence[int], int]): dimension of input image.
            patch_size (Union[Sequence[int], int]): dimension of patch size.
            hidden_size (int, optional): dimension of hidden layer. Defaults to 768.
            mlp_dim (int, optional): dimension of feedforward layer. Defaults to 3072.
            num_layers (int, optional): number of transformer blocks. Defaults to 12.
            num_heads (int, optional): number of attention heads. Defaults to 12.
            proj_type (str, optional): patch embedding layer type. Defaults to "conv".
            pos_embed_type (str, optional): position embedding type. Defaults to "learnable".
            classification (bool, optional): bool argument to determine if classification is used. Defaults to False.
            num_classes (int, optional): number of classes if classification is used. Defaults to 2.
            dropout_rate (float, optional): fraction of the input units to drop. Defaults to 0.0.
            spatial_dims (int, optional): number of spatial dimensions. Defaults to 3.
            post_activation (str, optional): add a final acivation function to the classification head
                when `classification` is True. Default to "Tanh" for `nn.Tanh()`.
                Set to other values to remove this function.
            qkv_bias (bool, optional): apply bias to the qkv linear layer in self attention block. Defaults to False.
            save_attn (bool, optional): to make accessible the attention in self attention block. Defaults to False.

        Examples::

            # for single channel input with image size of (96,96,96), conv position embedding and segmentation backbone
            >>> net = ViT(in_channels=1, img_size=(96,96,96), proj_type='conv', pos_embed_type='sincos')

            # for 3-channel with image size of (128,128,128), 24 layers and classification backbone
            >>> net = ViT(in_channels=3, img_size=(128,128,128), proj_type='conv', pos_embed_type='sincos', classification=True)

            # for 3-channel with image size of (224,224), 12 layers and classification backbone
            >>> net = ViT(in_channels=3, img_size=(224,224), proj_type='conv', pos_embed_type='sincos', classification=True,
            >>>           spatial_dims=2)

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.classification = classification
        self.patch_embedding = PatchEmbeddingBlockMonai(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            proj_type=proj_type,
            pos_embed_type=pos_embed_type,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size, mlp_dim, num_heads, dropout_rate, qkv_bias, save_attn
                )
                for i in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_size)
        if self.classification:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
            if post_activation == "Tanh":
                self.classification_head = nn.Sequential(
                    nn.Linear(hidden_size, num_classes), nn.Tanh()
                )
            else:
                self.classification_head = nn.Linear(hidden_size, num_classes)  # type: ignore

    def forward(self, x):
        x = self.patch_embedding(x)
        if hasattr(self, "cls_token"):
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
        hidden_states_out = []
        for blk in self.blocks:
            x = blk(x)
            hidden_states_out.append(x)
        x = self.norm(x)
        if hasattr(self, "classification_head"):
            x = self.classification_head(x[:, 0])
        return x, hidden_states_out


class ViT(nn.Module):
    def __init__(
        self,
        posenc: PosEncType,
        img_size: Sequence[int] | int,
        patch_embed_type: PatchEmbeddingType = PatchEmbeddingType.CONV,
        num_classes: int = 2,
        scale: int = 1,
        temperature: int = 10000,
        patch_size: Sequence[int] | int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 8,
        num_heads: int = 8,
        dropout_rate: float = 0.0,
        variance_factors=None,
    ):
        super().__init__()
        self.posenc = posenc

        self.vit = ViTMonai(
            in_channels=1,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            proj_type="conv",
            pos_embed_type="learnable",
            classification=True,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            spatial_dims=2,
            post_activation="none",
            qkv_bias=False,
            save_attn=False,
        )

        self.vit.patch_embedding = PatchEmbeddingBlock(
            pos_embed_type=posenc,
            patch_embed_type=patch_embed_type,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            in_channels=1,
            spatial_dims=2,
            scale=scale,
            temperature=temperature,
            variance_factors=variance_factors,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vit(x)


class ViTLucid(nn.Module):
    def __init__(
        self,
        posenc: PosEncType,
        img_size: Sequence[int] | int = 224,
        patch_embed_type: PatchEmbeddingType = PatchEmbeddingType.CONV,
        num_classes: int = 2,
        scale: int = 1,
        temperature: int = 10000,
        patch_size: Sequence[int] | int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 8,
        num_heads: int = 8,
        dropout_rate: float = 0.1,
        variance_factors=None,
    ):
        super().__init__()
        self.posenc = posenc

        spatial_dims = 2
        channels = 1

        image_height, image_width = pair(img_size, spatial_dims)
        patch_height, patch_width = pair(patch_size, spatial_dims)

        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."

        patch_dim = channels * patch_height * patch_width
        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, hidden_size),
            nn.LayerNorm(hidden_size),
        )

        self.pos_embedding = PositionalEmbedding(
            posenc,
            img_size,
            patch_size,
            hidden_size,
            spatial_dims,
            n_tokens=1,
            scale=scale,
            temperature=temperature,
            variance_factors=variance_factors,
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))

        self.dropout = nn.Dropout(dropout_rate)

        self.transformer = Transformer(
            hidden_size, num_layers, num_heads, 64, mlp_dim, dropout_rate
        )

        self.cls_head = nn.Linear(hidden_size, num_classes)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b = x.shape[0]

        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)

        x = torch.cat((cls_tokens, x), dim=1)

        x = self.pos_embedding(x)

        x = self.dropout(x)

        x = self.transformer(x)

        return self.cls_head(x[:, 0])


class VideoVisionTransformer(nn.Module):
    def __init__(
        self,
        posenc_type,
        image_size: int,
        patch_size_spatial: int,
        patch_size_temporal: int,
        num_frames: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        num_classes: int = 1,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        temperature=10e4,
        scale=1.0,
        variance_factors=None,
    ):
        super().__init__()

        assert (
            image_size % patch_size_spatial == 0
        ), "Image size must be divisible by the spatial patch size."
        assert (
            num_frames % patch_size_temporal == 0
        ), "Number of frames must be divisible by the temporal patch size."

        self.hidden_dim = hidden_dim
        num_patches_spatial = (image_size // patch_size_spatial) ** 2
        num_patches_temporal = num_frames // patch_size_temporal
        num_patches = num_patches_spatial * num_patches_temporal

        # 3D Patch Embedding Layer (Spatial + Temporal)
        self.patch_embed = nn.Conv3d(
            in_channels=1,
            out_channels=hidden_dim,
            kernel_size=(patch_size_temporal, patch_size_spatial, patch_size_spatial),
            stride=(patch_size_temporal, patch_size_spatial, patch_size_spatial),
        )

        # Class Token
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        # Positional Encoding
        self.cls_positional_encoding = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.positional_encoding = PositionalEmbedding(
            posenc_type,
            [num_frames, image_size, image_size],
            [patch_size_temporal, patch_size_spatial, patch_size_spatial],
            hidden_dim,
            3,
            temperature,
            scale,
            variance_factors=variance_factors,
        )

        # Transformer Encoder with batch_first=True
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,  # Use batch_first=True to suppress the warning
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        # Classification Head
        self.head = nn.Sequential(
            norm_layer(hidden_dim),
            nn.Linear(hidden_dim, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.cls_positional_encoding, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.head[1].bias.data[0] = 55.6

    def forward(self, x: torch.Tensor):
        # Input shape: [B, C, T, H, W]
        x = rearrange(x, "b t c h w -> b c t h w")
        x = self.patch_embed(x)  # [B, hidden_dim, T', H', W']
        x = rearrange(x, "b c t h w -> b (t h w) c")  # [B, num_patches, hidden_dim]

        # Add class token and positional encoding
        batch_size = x.shape[0]
        cls_token = (self.class_token + self.cls_positional_encoding).expand(
            batch_size, -1, -1
        )

        x = torch.cat([cls_token, self.positional_encoding(x)], dim=1)

        # Pass through Transformer encoder
        x = self.encoder(x)  # [B, num_patches+1, hidden_dim]

        # Use the class token for classification
        cls_output = x[:, 0]  # [B, hidden_dim]

        # Pass through the head
        return self.head(cls_output)
