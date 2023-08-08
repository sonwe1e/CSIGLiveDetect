# dataset for SuHiFiMask
import os
import random
import cv2
import numpy as np
from torch.utils.data import DataLoader
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from tqdm import tqdm


def read_data(name_label, image_path, split="train"):
    if split != "test":
        image_name, label = name_label.split(" ")
        return image_name, cv2.imread(image_path + image_name), int(label)
    else:
        image_name = name_label.split("\n")[0]
        return image_name, cv2.imread(image_path + image_name), None


class SuHiFiMaskDataset(Dataset):
    def __init__(self, split="train", transform=None, fold=0, max_threads=24):
        self.split = split
        self.transform = transform
        self.labels = {}
        self.images = {}
        self.images_list = []
        self.data_path = "./phase1/"
        if split != "test":
            train_list = open("./data_info/train_label.txt", "r").readlines()
            dev_list = open("./data_info/dev_label.txt", "r").readlines()
            random.shuffle(train_list)
            random.shuffle(dev_list)
            self.name_label = train_list + dev_list
            # random.shuffle(self.name_label)
            l = len(self.name_label) // 6
            self.name_label = (
                (self.name_label[: fold * l] + self.name_label[(fold + 1) * l :])
                if split == "train"
                else self.name_label[fold * l : (fold + 1) * l]
            )
        else:
            self.name_label = open(f"./data_info/{self.split}.txt", "r").readlines()
        # with ThreadPoolExecutor(max_workers=max_threads) as executor:
        #     read_data_partial = partial(
        #         read_data, image_path=self.data_path, split=split
        #     )
        #     for image_name, image, label in tqdm(
        #         executor.map(read_data_partial, self.name_label)
        #     ):
        #         self.images[image_name] = image
        #         self.labels[image_name] = label
        #         self.images_list.append(image_name)

    def __len__(self):
        return len(self.name_label)

    def __getitem__(self, idx):
        image_name = self.name_label[idx]
        image_name, image, label = read_data(
            image_path=self.data_path, name_label=image_name, split=self.split
        )
        if self.split != "test":
            if self.split == "train" and np.random.rand() < 0.3:
                image_name2 = self.name_label[
                    np.random.randint(0, len(self.name_label))
                ]
                image_name2, image2, label2 = read_data(
                    image_path=self.data_path, name_label=image_name2, split=self.split
                )
                image2 = cv2.resize(image2, (image.shape[1], image.shape[0]))
                lam = np.random.beta(0.2, 0.2)
                image = lam * image + (1 - lam) * image2
                image = image.astype(np.uint8)
                label = lam * label + (1 - lam) * label2
            label = torch.tensor(int(label)).float()
            if self.transform:
                image = self.transform(image=image)["image"]
            return image, label.unsqueeze(0)
        else:
            image = self.images[image_name]
            if self.transform:
                image = self.transform(image=image)["image"]
            return image, image_name


# 训练集和验证集的数据增强
train_transform = A.Compose(
    [
        A.RandomResizedCrop(
            224,
            224,
            scale=(0.64, 1.0),
            p=1,
        ),
        A.Flip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.HueSaturationValue(p=0.3),
        A.GaussNoise(p=0.3),
        A.Blur(p=0.3),
        A.MotionBlur(p=0.3),
        A.Sharpen(p=0.3),
        A.ImageCompression(p=0.3),
        A.ColorJitter(p=0.3),
        A.Normalize(),
        ToTensorV2(),
    ]
)
test_transform = A.Compose(
    [
        A.Resize(224, 224),
        A.Normalize(),
        ToTensorV2(),
    ]
)


def get_dataloader(split="train", bs=512, transform=train_transform, fold=1):
    dataset = SuHiFiMaskDataset(
        split=split,
        transform=transform,
        fold=fold,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=bs,
        shuffle=True if split == "train" else False,
        num_workers=8,
        persistent_workers=True,
        pin_memory=True,
    )
    return dataloader


if __name__ == "__main__":
    train_loader = get_dataloader("train", fold=0)

    for batch in train_loader:
        print(batch[0].shape)
        print(batch[1].shape)
        break
    dev_loader = get_dataloader("dev", fold=0)
    for batch in dev_loader:
        print(batch[0].shape)
        print(batch[1].shape)
        break
    test_loader = get_dataloader("test", fold=0)
    for batch in test_loader:
        print(batch[0].shape)
        break
