import cv2
import numpy as np
from torch.utils.data import DataLoader
import torch
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from tqdm import tqdm


def read_data(name_label, image_path, split="test"):
    if split != "test":
        image_name, label = name_label.split(" ")
        return (
            image_name,
            cv2.imread(image_path + image_name),
            torch.tensor(int(label)).float(),
        )
    else:
        image_name = name_label.split("\n")[0]
        return (
            image_name,
            cv2.imread(image_path + image_name),
            None,
        )


train_transform = A.Compose(
    [
        A.Resize(224, 224),
        A.Normalize(),
        ToTensorV2(),
    ]
)

valid_transform = A.Compose(
    [
        A.Resize(224, 224),
        A.Normalize(),
        ToTensorV2(),
    ]
)


class SuHiFiMaskDataset(Dataset):
    def __init__(self, split="train", transform=None, fold=1, max_threads=24):
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
            self.name_label = train_list if split == "train" else dev_list
            # self.name_label = train_list + dev_list
            # l = len(self.name_label) // 5
            # self.name_label = (
            #     (self.name_label[: fold * l] + self.name_label[(fold + 1) * l :])
            #     if split == "train"
            #     else self.name_label[fold * l : (fold + 1) * l]
            # )
        else:
            self.name_label = open(f"./data_info/{self.split}.txt", "r").readlines()
        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            read_data_partial = partial(
                read_data, image_path=self.data_path, split=split
            )
            for image_name, image, label in tqdm(
                executor.map(read_data_partial, self.name_label)
            ):
                self.images[image_name] = image
                self.labels[image_name] = label
                self.images_list.append(image_name)

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        image_name = self.images_list[idx]
        if self.split != "test":
            image, label = self.images[image_name], self.labels[image_name]
            if self.transform:
                image = self.transform(image=image)["image"]
            return image, label.unsqueeze(0)
        else:
            image = self.images[image_name]
            if self.transform:
                image = self.transform(image=image)["image"]
            return image, image_name


def get_dataloader(split="train", bs=512, transform=None, fold=1):
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
        pin_memory=True,
        persistent_workers=True,
    )
    return dataloader


if __name__ == "__main__":
    train_loader = get_dataloader("train", fold=0)

    for batch in train_loader:
        print(batch[0], batch[1])
        break
