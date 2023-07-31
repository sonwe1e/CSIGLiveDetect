import cv2
import numpy as np
from torch.utils.data import DataLoader
import torch
import random
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os


def mixup(image1, image2, label1, label2, alpha=0.25):
    """
    实现Mixup数据增强技术
    :param image1: 第一个图像，格式为 (224, 224, 3)
    :param label1: 第一个图像对应的标签
    :param image2: 第二个图像，格式为 (224, 224, 3)
    :param label2: 第二个图像对应的标签
    :param alpha: 超参数alpha用于控制Mixup的强度，取值范围为 [0, 1]
    :return: 混合后的图像和对应的标签
    """
    assert 0.0 <= alpha <= 1.0, "alpha超出取值范围 [0, 1]"

    # 随机生成lambda值，lambda值用于确定混合的比例
    lambda_ = np.random.beta(alpha, alpha)

    # 使用lambda值进行线性插值
    mixed_image = lambda_ * image1 + (1 - lambda_) * image2

    # 使用lambda值进行标签插值
    mixed_label = lambda_ * label1 + (1 - lambda_) * label2

    return mixed_image, mixed_label


def cutmix(image1, image2, label1, label2, alpha=0.25):
    """
    实现CutMix数据增强技术
    :param image1: 第一个图像，格式为 (224, 224, 3)
    :param image2: 第二个图像，格式为 (224, 224, 3)
    :param alpha: 超参数alpha用于控制CutMix的强度，取值范围为 [0, 1]
    :return: 混合后的图像和对应的标签
    """
    assert 0.0 <= alpha <= 1.0, "alpha超出取值范围 [0, 1]"

    # 随机生成lambda值，lambda值用于确定混合的比例
    lambda_ = np.random.beta(alpha, alpha)

    # 随机生成混合的区域
    cut_w = int(image1.shape[1] * lambda_)
    cut_h = int(image1.shape[0] * lambda_)
    cx = np.random.randint(0, image1.shape[1] - cut_w + 1)
    cy = np.random.randint(0, image1.shape[0] - cut_h + 1)

    # 创建新的图像
    new_image = image1.copy()
    new_image[cy : cy + cut_h, cx : cx + cut_w, :] = image2[
        cy : cy + cut_h, cx : cx + cut_w, :
    ]
    new_label = lambda_ * label1 + (1 - lambda_) * label2

    return new_image, new_label


def read_data(name_label, image_path, split="test"):
    if split != "test":
        image_name, label = name_label.split(" ")
        return image_name, cv2.imread(image_path + image_name), int(label)
    else:
        image_name = name_label.split("\n")[0]
        return image_name, cv2.imread(image_path + image_name), None


class SuHiFiMaskDataset(Dataset):
    def __init__(self, split="train", transform=None, fold=0, max_threads=32):
        self.split = split
        self.transform = transform
        self.labels = {}
        self.images = {}
        self.images_list = []
        self.data_path = "./phase1/"
        if split != "test":
            self.name_label = (
                open(f"./data_info/train_label.txt", "r").readlines()
                + open(f"./data_info/dev_label.txt", "r").readlines()
            )
            random.shuffle(self.name_label)
            l = len(self.name_label) // 6
            self.name_label = (
                (self.name_label[: fold * l] + self.name_label[(fold + 1) * l :])
                if split == "train"
                else self.name_label[fold * l : (fold + 1) * l]
            )
        else:
            self.name_label = open(f"./data_info/{self.split}.txt", "r").readlines()

    def __len__(self):
        return len(self.name_label)

    def __getitem__(self, idx):
        image_name = self.name_label[idx]
        image_name, image, label = read_data(
            name_label=image_name, image_path=self.data_path, split=self.split
        )
        if self.split != "test":
            if self.split == "train" and np.random.rand() < 0.5:
                if np.random.rand() < 0.4:
                    idx2 = np.random.randint(0, len(self.name_label))
                    image_name2 = self.name_label[idx2]
                    image_name2, image2, label2 = read_data(
                        name_label=image_name2,
                        image_path=self.data_path,
                        split=self.split,
                    )
                    image2 = cv2.resize(image2, (image.shape[1], image.shape[0]))
                    image, label = cutmix(image, image2, label, label2)
                if np.random.rand() < 0.4:
                    idx2 = np.random.randint(0, len(self.name_label))
                    image_name2 = self.name_label[idx2]
                    image_name2, image2, label2 = read_data(
                        name_label=image_name2,
                        image_path=self.data_path,
                        split=self.split,
                    )
                    image2 = cv2.resize(image2, (image.shape[1], image.shape[0]))
                    image, label = mixup(image, image2, label, label2)
            image = image.astype(np.uint8)
            label = torch.tensor(label).float()
            if self.transform:
                image = self.transform(image=image)["image"]
            return image, label.unsqueeze(0)
        else:
            if self.transform:
                image = self.transform(image=image)["image"]
            return image, image_name


class SuHiFiMaskDataset_adv(Dataset):
    def __init__(self, split="train", transform=None, fold=0):
        super().__init__()
        self.data_path = "./phase2/"
        self.image_list = os.listdir(self.data_path)
        self.transform = transform
        random.shuffle(self.image_list)
        l = len(self.image_list) // 6
        self.image_list = (
            self.image_list[: fold * l] + self.image_list[(fold + 1) * l :]
            if split == "train"
            else self.image_list[fold * l : (fold + 1) * l]
        )

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_name = self.image_list[index]
        image = cv2.imread(self.data_path + self.image_name)
        label = torch.tensor(int(image_name[-5]))
        if random.random() < 0.5:
            index2 = np.random.randint(0, len(self.image_list))
            image_name2 = self.image_list[index2]
            image2 = cv2.imread(self.data_path + self.image_name2)
            label2 = torch.tensor(int(image_name2[-5]))
            image2 = cv2.resize(image2, (image.shape[1], image.shape[0]))
            if random.random() < 0.5:
                image, label = mixup(image, image2, label, label2)
            if random.random() < 0.5:
                image, label = cutmix(image, image2, label, label2)
        image = image.astype(np.uint8)
        if self.transform:
            image = self.transform(image=image)["image"]
        return image, label.unsqueeze(0)


# 训练集和验证集的数据增强
train_transform = A.Compose(
    [
        A.RandomResizedCrop(
            224,
            224,
            scale=(0.2, 1.0),
            p=1,
        ),
        A.HorizontalFlip(p=0.5),
        A.OpticalDistortion(p=0.3),
        A.ShiftScaleRotate(p=0.3),
        A.Defocus(p=0.3),
        A.RandomBrightnessContrast(p=0.3),
        A.HueSaturationValue(p=0.3),
        A.GaussNoise(p=0.3),
        A.AdvancedBlur(p=0.3),
        A.MotionBlur(p=0.3),
        A.Sharpen(p=0.3),
        A.ImageCompression(p=0.3),
        A.ISONoise(p=0.3),
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


def get_dataloader(
    split="train", bs=512, transform=train_transform, fold=1, type="phase1"
):
    dataset = (
        SuHiFiMaskDataset(
            split=split,
            transform=transform,
            fold=fold,
        )
        if type == "phase1"
        else SuHiFiMaskDataset_adv(
            split=split,
            transform=transform,
            fold=fold,
        )
    )
    dataloader = DataLoader(
        dataset,
        batch_size=bs,
        shuffle=True if split == "train" else False,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True,
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
