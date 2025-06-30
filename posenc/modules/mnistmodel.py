from typing import Iterable

import lightning.pytorch as L
import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from medmnist import INFO
from torchmetrics import AUROC, Accuracy, MetricCollection
from torchvision.models.video import r3d_18

from posenc.enums import PosEncType
from posenc.nets.positional_encodings import PositionalEmbedding

# Bias is defined based on training data class imbalance. bias = log(class_0 / class_1) for binary classification.
SHAPE_INIT_BIAS = {
    "vesselmnist3d": torch.tensor(-2.0669),
    "adrenalmnist3d": torch.tensor(-1.2773)
}

def accuracy(pred, target, binary=True):
    if binary:
        pred = torch.sigmoid(pred)
        pred = (pred > 0.5).float()
    else:
        pred = torch.argmax(pred, dim=1)
    return (pred == target).float().mean()

def pair(t):
    return t if isinstance(t, Iterable) else (t, t, t)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    def __init__(self, *, image_size, image_patch_size, num_classes, dim,
                 depth, heads, mlp_dim, pool = 'cls', channels = 1, spatial_dims=3, dim_head = 64,
                 dropout = 0., emb_dropout = 0., pos_embed_type = PosEncType.ISOFPE, variance_factors = None):
        super().__init__()
        image_height, image_width, image_depth = pair(image_size)
        patch_height, patch_width, patch_depth = pair(image_patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0 and image_depth % patch_depth == 0, 'Image dimensions must be divisible by the patch size.'

        patch_dim = channels * patch_height * patch_width * patch_depth

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) (d p3) -> b (h w d) (p1 p2 p3 c)', p1 = patch_height, p2 = patch_width, p3 = patch_depth),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = PositionalEmbedding(
            pos_embed_type=pos_embed_type,
            img_size=image_size,
            patch_size=image_patch_size,
            hidden_size=dim,
            spatial_dims=spatial_dims,
            variance_factors=variance_factors,
            n_tokens=1
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, video):
        x = self.to_patch_embedding(video)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding(x)
        
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)

class ShapeM(nn.Module):
    def __init__(self, *, image_size, image_patch_size, num_classes, dim, depth, heads, mlp_dim, 
                 pool='cls', dim_head=64, dropout=0., emb_dropout=0., 
                 pos_embed_type=PosEncType.ISOFPE, variance_factors=None):
        super().__init__()

        image_height, image_width, image_depth = pair(image_size)
        patch_height, patch_width, patch_depth = pair(image_patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0 and image_depth % patch_depth == 0, 'Image dimensions must be divisible by the patch size.'
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # Use R3D_18 as feature extractor
        self.feature_extractor = r3d_18()
        
        # Modify first layer to accept single channel input
        self.feature_extractor.stem[0] = nn.Conv3d(1, 64, kernel_size=(7, 7, 7), 
                                                 stride=(2, 2, 2), padding=(3, 3, 3), 
                                                 bias=False)
        
        # Trim feature extractor to remove the last few layers. Halfs the input size.
        self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-5])
        
        # Get feature map dimensions
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, *image_size)
            features = self.feature_extractor(dummy_input)
            self.feature_shape = features.shape[2:]  # D, H, W
            self.feature_channels = features.shape[1]
        
        self.to_patch_embedding = nn.Sequential(
            nn.Conv3d(self.feature_channels, dim, kernel_size=(patch_height, patch_width, patch_depth), stride=(patch_height, patch_width, patch_depth)),
            Rearrange('b c d h w -> b (d h w) c'),  # Rearrange to sequence of patches
            nn.LayerNorm(dim),
        )
        
        # Positional embedding
        self.pos_embedding = PositionalEmbedding(
            pos_embed_type=pos_embed_type,
            img_size=self.feature_shape,
            patch_size=image_patch_size,
            hidden_size=dim,
            spatial_dims=3,
            variance_factors=variance_factors,
            n_tokens=1
        )
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        
        # Small transformer
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        
        # Pooling and classification
        self.pool = pool
        self.to_latent = nn.Identity()
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
    
    def forward(self, x):
        # Extract features with ResNet3D
        x = self.feature_extractor(x)
        
        # Rearrange to sequence of spatial features
        x = self.to_patch_embedding(x)
        
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_embedding(x)

        x = self.dropout(x)

        # Process with transformer
        x = self.transformer(x)
        
        # Pool and classify
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        return self.mlp_head(x) 


class MNISTModel(L.LightningModule):
    def __init__(self, flag: str, lr=0.001, image_size=64, image_patch_size=4,
                  dropout=0.1, emb_dropout=0.1, weight_decay=0.01,
                  pos_emb_type=PosEncType.ISOFPE, variance_factors=None):
        super(MNISTModel, self).__init__()

        self.lr = lr
        self.weight_decay = weight_decay 

        self.info = INFO[flag]
        self.task = self.info["task"]
        self.binary = self.task == "binary-class"

        self.num_classes = 1 if self.binary else len(self.info["label"])

        self.save_hyperparameters()

        if flag.lower() in ["vesselmnist3d", "adrenalmnist3d"]:
            # For 3D shape datasets, we use the ShapeM model
            self.model = ShapeM(
                image_size=image_size,
                image_patch_size=image_patch_size,
                num_classes=self.num_classes,
                dim=144,
                depth=1,
                heads=1,
                mlp_dim=288,
                pool='cls',
                dropout=dropout,
                emb_dropout=emb_dropout,
                pos_embed_type=pos_emb_type,
                variance_factors=variance_factors
            )
            with torch.no_grad():
                self.model.mlp_head[-1].bias.copy_(SHAPE_INIT_BIAS[flag.lower()])
        else:
            self.model = ViT(
                image_size = image_size,          
                image_patch_size = image_patch_size,    
                num_classes = self.num_classes,
                dim = 144,
                depth = 2,
                heads = 4,
                mlp_dim = 288,
                dropout = dropout,
                emb_dropout = emb_dropout,
                pos_embed_type = pos_emb_type,
                variance_factors = variance_factors,
            )

        if self.binary:
            self.criterion = nn.BCEWithLogitsLoss()
            self.process_label = lambda x: x.flatten().to(torch.float32)
            self.activation = nn.Sigmoid()
            self.metrics = MetricCollection({
                "accuracy": Accuracy(task="binary"),
                "auroc": AUROC(task="binary")
            })
        else:
            self.criterion = nn.CrossEntropyLoss()
            self.process_label = lambda x: x.flatten()
            self.activation = nn.Softmax(dim=1)
            self.metrics = MetricCollection({
                "accuracy": Accuracy(task="multiclass", num_classes=self.num_classes),
                "auroc": AUROC(task="multiclass", num_classes=self.num_classes)
            })

    def forward(self, x):
        pred = self.model(x)
        if self.binary:
            # For binary classification, just squeeze the last dimension if needed
            return pred.squeeze(-1)
        else:
            return pred

    def step(self, batch):
        img, label = batch[0], batch[1]
        label = self.process_label(label)
        pred = self(img)

        return pred, label


    def training_step(self, batch, batch_idx):
        pred, label = self.step(batch)

        loss = self.criterion(pred, label)
        self.log("train_loss", loss, prog_bar=True, logger=True, on_epoch=True, on_step=True)
    
        return loss
    
    def validation_step(self, batch, batch_idx):

        pred, label = self.step(batch)

        loss = self.criterion(pred, label)
        performance = self.metrics(pred, label)
        
        self.log("val_loss", loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log_dict(performance, prog_bar=True, logger=True, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
    

