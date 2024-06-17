import os

import clip
import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import CIFAR100
from torchvision.io import read_image

from .paths import DATA_DIR

_, preprocess = clip.load("RN50")


class CIFAR100DataModule(L.LightningDataModule):
    def __init__(self, batch_size=512, num_workers=4, data_dir: str = DATA_DIR):
        super().__init__()
        self.data_dir = data_dir
        self.transform = preprocess
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        # download
        CIFAR100(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        # note that the split ratio here is odd.
        # this is because we don't actually train and validate
        # we just need a large validation set to compute class separation
        if stage == "fit":
            cifar_full = CIFAR100(self.data_dir, train=True, transform=self.transform)
            self.cifar_train, self.cifar_val = random_split(
                cifar_full,
                [25000, 25000],
                generator=torch.Generator().manual_seed(1234),
            )

        if stage == "test":
            self.cifar_test = CIFAR100(
                self.data_dir, train=False, transform=self.transform
            )

        if stage == "predict":
            self.cifar_predict = CIFAR100(
                self.data_dir, train=False, transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.cifar_train,
            shuffle=True,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
        )

    def val_dataloader(self):
        return DataLoader(
            self.cifar_val,
            shuffle=False,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
        )

    def test_dataloader(self):
        return DataLoader(
            self.cifar_test,
            shuffle=False,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
        )


class CustomCOCODataModule(L.LightningDataModule):
    def __init__(
        self,
        token_type: str,
        layer_no: int,
        representation_dir=f"{DATA_DIR}/meta-llama/Meta-Llama-3-70B",
        train_dir=f"{DATA_DIR}/coco/images/train2017",
        val_dir=f"{DATA_DIR}/coco/images/val2017",
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.representation_dir = representation_dir
        self.token_type = token_type
        self.layer_no = layer_no
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = preprocess
        self.target_dir = f"{self.representation_dir}/{self.token_type}"

        self.save_hyperparameters("layer_no", "token_type")

    def setup(self, stage):
        if stage in (None, "fit"):
            self.train_dataset = CustomCOCODataset(
                self.train_dir, self.target_dir, self.layer_no, self.transform
            )
            self.val_dataset = CustomCOCODataset(
                self.val_dir, self.target_dir, self.layer_no, self.transform
            )
        elif stage == "test":
            self.test_dataset = CustomCOCODataset(
                self.val_dir, self.target_dir, self.layer_no, self.transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


class CustomCOCODataset(Dataset):
    def __init__(self, img_dir: str, target_dir: str, target_row: int, transform=None):
        self.img_dir = img_dir
        self.target_dir = target_dir
        self.target_row = target_row
        self.transform = transform
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith(".jpg")]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = read_image(img_path)

        target_name = img_name.replace(".jpg", ".pt").rstrip("0")
        target_path = os.path.join(self.target_dir, target_name)
        target_tensor = torch.load(target_path)[self.target_row]

        if self.transform:
            img = self.transform(img)

        return img, target_tensor


# Example usage:
# data_module = CustomCOCODataModule(
#     train_dir='/path/to/train2017',
#     val_dir='/path/to/val2017',
#     target_dir='/path/to/targets',
#     target_row=0,
#     batch_size=32,
#     num_workers=4
# )
