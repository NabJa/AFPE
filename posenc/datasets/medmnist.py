import lightning.pytorch as L
import medmnist
import numpy as np
from medmnist import INFO
from monai.transforms import Resize
from torch.utils.data import DataLoader

data_flags = ['organmnist3d', 'nodulemnist3d', 'adrenalmnist3d', 'fracturemnist3d', 'vesselmnist3d', 'synapsemnist3d']


class AnistropicDataset:
    def __init__(self, root, flag, split, anisotropy=(1, 1, 3), size=64, interpolate=True):
        self.anisotropy = anisotropy
        self.interpolate = interpolate
        
        self.info = INFO[flag]
        DataClass = getattr(medmnist, self.info['python_class'])

        self.dataset = DataClass(root=root, split=split, size=size)

        if self.interpolate:
            self.resize = Resize([size, size, size])
        else:
            self.resize = lambda x: x

        self.anisotropy = anisotropy

        self.is_unchanged = (np.array(anisotropy) == 1).all()
        self.is_isotrop = len(np.unique(anisotropy)) == 1
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):        
        img, label = self.dataset[index]
        img = img.astype(np.float32)

        if self.is_unchanged:
            return img, label
        
        img = self.resize(img[:, ::self.anisotropy[0], ::self.anisotropy[1], ::self.anisotropy[2]])
        return img, label

def get_dataset(root, flag, anisotropy=(1, 1, 1), interpolate=True):
    
    train = AnistropicDataset(root, flag, split="train", anisotropy=anisotropy, size=64, interpolate=interpolate)
    valid = AnistropicDataset(root, flag, split="val", anisotropy=anisotropy, size=64, interpolate=interpolate)
    test =  AnistropicDataset(root, flag, split="test", anisotropy=anisotropy, size=64, interpolate=interpolate)
    
    return train, valid, test


class MNISTDataModule(L.LightningDataModule):
    def __init__(self, root, flag, anisotropy=(1, 1, 1), batch_size=8, num_workers=8, persistent_workers=True, interpolate=True):
        
        super(MNISTDataModule, self).__init__()
        self.root = root
        self.flag = flag
        self.anisotropy = anisotropy
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.interpolate = interpolate

    def setup(self, stage=None):
        self.train, self.valid, self.test = get_dataset(self.root, self.flag, self.anisotropy, self.interpolate)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers,
                           shuffle=True, pin_memory=True, persistent_workers=self.persistent_workers)
    
    def val_dataloader(self):
        return DataLoader(self.valid, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers,shuffle=False)
