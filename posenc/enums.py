from enum import Enum


class ViTSettings:
    """Defenitions as in https://arxiv.org/abs/2106.10270"""

    def __init__(self, variation: str):
        self.settings_map = {
            "vit-shallow": {"mlp_dim": 768, "num_layers": 3, "num_heads": 6},
            "vit-t": {"mlp_dim": 192, "num_layers": 12, "num_heads": 3},
            "vit-s": {"mlp_dim": 384, "num_layers": 12, "num_heads": 6},
            "vit-b": {"mlp_dim": 768, "num_layers": 12, "num_heads": 12},
            "vit-u": {"mlp_dim": 3072, "num_layers": 12, "num_heads": 12},
            "vit-l": {"mlp_dim": 1024, "num_layers": 24, "num_heads": 16},
        }
        if variation.lower() in self.settings_map:
            settings = self.settings_map[variation]
            self.mlp_dim = settings["mlp_dim"]
            self.num_layers = settings["num_layers"]
            self.num_heads = settings["num_heads"]
        else:
            raise ValueError(f"ViT variation {variation} not found in settings.")

    def __str__(self):
        return f"ViTSettings(mlp_dim={self.mlp_dim}, num_layers={self.num_layers}, num_heads={self.num_heads})"

    def __repr__(self):
        return self.__str__()


class ModelType(Enum):
    VIT_SHALLOW = "vit-shallow"
    VIT_T = "vit-t"
    VIT_S = "vit-s"
    VIT_B = "vit-b"
    VIT_U = "vit-u"
    VIT_L = "vit-l"
    CNN = "cnn"


class DataTaskType(Enum):
    CHESTX_MULTI = "chestxmulti"
    ECHONET_REG = "echonetreg"

class PosEncType(Enum):
    SINCOS = "sincos"  # Sinusoidal Positional Encoding
    FOURIER = "fourier"  # Fourier Positional Encoding
    ISOFPE = "isofpe"  # Isotropic Fourier Positional Encoding
    LFPE = "lfpe"  # Learned Fourier Positional Encoding
    LEARNABLE = "learnable"  # Learnable Positional Encoding from normal distribution initialization
    NONE = "none"  # No positional encoding
    LSFPE = "lsfpe"  # Learned Scale Fourier Positional Encoding


class PatchEmbeddingType(Enum):
    CONV = "conv"
    VIDEO = "video"
    VIDEOV2 = "videov2"


class OptimizerType(Enum):
    ADAMW = "adamw"
    SGD = "sgd"


class SchedulerType(Enum):
    COSINE = "cosine"
    WARMUPCOSINE = "warmupcosine"
    WARMUPEXP = "warmupexp"


class PredictionTask(Enum):
    MULTILABEL = "multilabel"
    REGRESSION = "regression"
