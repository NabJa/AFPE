import sys
from pathlib import Path

import torch

# Add the parent directory of 'notebooks' to sys.path
sys.path.append(str(Path('.').resolve().parent))

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from medmnist import INFO

from posenc.datasets.medmnist import MNISTDataModule

# Now you can import the function
from posenc.enums import PosEncType
from posenc.modules.mnistmodel import MNISTModel

# # Default used by PyTorch
# torch.set_float32_matmul_precision("highest")
# Faster, but less precise
# torch.set_float32_matmul_precision("high")
# # Even faster, but also less precise
torch.set_float32_matmul_precision("medium")
seed_everything(42)


MNISTROOT = "/sc-scratch/sc-scratch-gbm-radiomics/medmnist"
FLAG = "adrenalmnist3d" # "organmnist3d" # "fracturemnist3d"

info = INFO[FLAG]

task = info["task"]

anisotopys = [(1, 1, 1), (1, 1, 4), (1, 1, 6), (1, 1, 8)]
posencs = [PosEncType.FOURIER, PosEncType.ISOFPE, PosEncType.SINCOS, PosEncType.NONE, PosEncType.LEARNABLE, PosEncType.LFPE]

for anisotropy in anisotopys:
    for posenc in posencs:

        save_dir = f"/sc-scratch/sc-scratch-gbm-radiomics/posenc/mnist/{FLAG}/{posenc.name}/anisotropy_{anisotropy[0]}_{anisotropy[1]}_{anisotropy[2]}"
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        dm = MNISTDataModule(
            MNISTROOT, flag=FLAG, anisotropy=anisotropy, interpolate=False,
            batch_size=32, num_workers=32, 
            # batch_size=8, num_workers=16, 
                             )

        image_size = torch.tensor([64, 64, 64])
        image_size = (image_size - 1) // torch.tensor(anisotropy) + 1

        # image_patch_size = torch.round(torch.tensor([4, 4, 4]) / torch.tensor(anisotropy)).clip(1).type(torch.int)
        image_patch_size = torch.round(torch.tensor([2, 2, 2]) / torch.tensor(anisotropy)).clip(1).type(torch.int)
        
        if posenc == PosEncType.FOURIER:
            posenc = PosEncType.ISOFPE
            variance_factors = [1, 1, 1]
        else:
            variance_factors = 0.5 / torch.tensor(anisotropy)

        model = MNISTModel(FLAG, lr=0.001, weight_decay=0.01, dropout=0.0, # weight_decay=0.001, # dropout=0.01,
                           pos_emb_type=posenc,
                           variance_factors=variance_factors,
                           image_size=image_size.tolist(), image_patch_size=image_patch_size.tolist())

        trainer = Trainer(max_epochs=75,
                        logger=CSVLogger(save_dir=save_dir),
                        callbacks=[ModelCheckpoint(monitor="val_loss", dirpath=save_dir)],
                        default_root_dir=save_dir,
                        )
        trainer.fit(model, datamodule=dm)
