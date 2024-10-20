import os

import lightning.pytorch as pl
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

# Custom dataset to load validation images without class subfolders
class ValImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        # List all image files in the folder
        self.image_filenames = [os.path.join(image_dir, fname) 
                                for fname in os.listdir(image_dir) 
                                if fname.endswith(('.jpg', '.jpeg', '.png'))]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = self.image_filenames[idx]
        # Open the image
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image


class SSLDataModule(pl.LightningDataModule):
    def __init__(
            self, 
            train_path, 
            val_path,
            img_size=224,
            batch_size=32,
            num_workers=4
            ):
        super().__init__()
        
        self.train_path = train_path
        self.val_path = val_path

        self.num_workers = num_workers

        self.img_size = img_size
        self.batch_size = batch_size

        self.train_transform, self.val_transform,  = self._get_transforms()


    def setup(self, stage=None):
        # Create an instance of the dataset
        self.train_dataset = datasets.ImageFolder(root=self.train_path, transform=self.train_transform)
        self.val_dataset = ValImageDataset(image_dir=self.val_path, transform=self.val_transform)


    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers
            )


    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers
            )


    def _get_transforms(self):
        mean = IMAGENET_DEFAULT_MEAN
        std = IMAGENET_DEFAULT_STD
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(self.img_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
        
        # eval transform
        t = []
        if self.img_size <= 224:
            crop_pct = 224 / 256
        else:
            crop_pct = 1.0
        size = int(self.img_size / crop_pct)
        t.append(
            transforms.Resize(size, interpolation=Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(self.img_size))
        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(mean=mean, std=std))
        
        val_transform = transforms.Compose(t)
        
        return train_transform, val_transform
