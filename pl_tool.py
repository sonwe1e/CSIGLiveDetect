import lightning.pytorch as pl
from dataset import *
import torch
import torch.nn as nn
from lion_pytorch import Lion
import numpy as np

torch.set_float32_matmul_precision("high")


class TeethSegment(pl.LightningModule):
    def __init__(self, config, model, len_train):
        super().__init__()
        self.config = config
        self.len_train = len_train
        self.learning_rate = config.learning_rate
        self.batch_size = config.batch_size
        self.model = model
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.train_predict = []
        self.train_label = []
        self.val_predict = []
        self.val_label = []

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        self.optimizer = Lion(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.learning_rate,
            epochs=self.config.epochs,
            steps_per_epoch=self.len_train,
            pct_start=0.06,
            final_div_factor=1e2,
            anneal_strategy="cos",
            cycle_momentum=False,
        )
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "interval": "step",
            },
        }

    def training_step(self, batch, batch_idx):
        x, y = batch
        tic = torch.rand(1)
        if tic < 0.3:
            x, y_a, y_b, lambda_ = self.mixup(x, y)
        elif 0.3 <= tic < 0.95:
            x, y_a, y_b, lambda_ = self.cutmix(x, y)
        else:
            x, y_a = self.random_erasing(x, y)
            y_b = y
            lambda_ = 1
        logits = self.model(x)
        bce_loss = lambda_ * (self.bce_loss(logits, y_a)) + (1 - lambda_) * (
            self.bce_loss(logits, y_b)
        )
        loss = bce_loss
        self.log("train_loss", loss, prog_bar=True)
        self.log("learning_rate", self.scheduler.get_last_lr()[0])
        self.train_predict.append(torch.sigmoid(logits))
        self.train_label.append(y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.bce_loss(logits, y)
        self.val_predict.append(torch.sigmoid(logits))
        self.val_label.append(y)
        self.log("valid_loss", loss)

    def on_train_epoch_end(self):
        self.train_predict = torch.cat(self.train_predict, dim=0)
        self.train_label = torch.cat(self.train_label, dim=0)
        apcer, bpcer = self.calculate_apcer_bpcer(self.train_predict, self.train_label)
        acer = (apcer + bpcer) / 2
        self.log("train_apcer", apcer, on_epoch=True)
        self.log("train_bpcer", bpcer, on_epoch=True)
        self.log("train_acer", acer, on_epoch=True, prog_bar=True)
        self.train_predict = []
        self.train_label = []

    def on_validation_epoch_end(self):
        self.val_predict = torch.cat(self.val_predict, dim=0)
        self.val_label = torch.cat(self.val_label, dim=0)
        apcer, bpcer = self.calculate_apcer_bpcer(self.val_predict, self.val_label)
        acer = (apcer + bpcer) / 2
        self.log("valid_apcer", apcer, on_epoch=True)
        self.log("valid_bpcer", bpcer, on_epoch=True)
        self.log("valid_acer", acer, on_epoch=True, prog_bar=True)
        self.val_predict = []
        self.val_label = []

    def mixup(self, images, labels, alpha=1):
        """
        在batch级别实现Mixup数据增强
        :param images: 一组图像，形状为 (B, C, H, W)
        :param labels: 对应的标签
        :param alpha: 控制Mixup的强度的超参数，取值范围为 [0, 1]
        :return: 混合后的图像和标签
        """
        if alpha > 0:
            lambda_ = np.random.beta(alpha, alpha)
        else:
            lambda_ = 1

        batch_size = images.size()[0]
        index = torch.randperm(batch_size, device=images.device)
        y_a, y_b = labels, labels[index]
        mixed_images = lambda_ * images + (1 - lambda_) * images[index]

        return mixed_images, y_a, y_b, lambda_

    def cutmix(self, images, labels, alpha=1):
        """
        在batch级别实现CutMix数据增强
        :param images: 一组图像，形状为 (B, C, H, W)
        :param labels: 对应的标签
        :param alpha: 控制CutMix的强度的超参数，取值范围为 [0, 1]
        :return: 混合后的图像和标签
        """
        if alpha > 0:
            lambda_ = np.random.beta(alpha, alpha)
        else:
            lambda_ = 1

        batch_size = images.size()[0]
        index = torch.randperm(batch_size, device=images.device)
        y_a, y_b = labels, labels[index]

        w, h = images.size(3), images.size(2)
        cut_ratio = np.sqrt(1 - lambda_)
        cut_w, cut_h = int(w * cut_ratio), int(h * cut_ratio)

        x1 = np.random.randint(0, w - cut_w + 1)
        y1 = np.random.randint(0, h - cut_h + 1)
        x2 = x1 + cut_w
        y2 = y1 + cut_h

        images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]

        mixed_images = images
        return mixed_images, y_a, y_b, lambda_

    def random_erasing(self, images, labels, sl=0.01, sh=0.16, r1=0.75, r2=1 / 0.75):
        """
        在batch级别实现Random Erasing数据增强
        :param images: 一组图像，形状为 (B, C, H, W)
        :param labels: 对应的标签
        :param probability: 擦除的概率
        :param sl: 擦除区域的最小面积比例
        :param sh: 擦除区域的最大面积比例
        :param r1: 长宽比的最小值
        :param r2: 长宽比的最大值
        :return: 经过随机擦除的图像和标签
        """
        batch_size, num_channels, img_height, img_width = images.size()

        for i in range(batch_size):
            area = img_height * img_width
            target_area = np.random.uniform(sl, sh) * area
            aspect_ratio = np.random.uniform(r1, r2)

            h = int(round(np.sqrt(target_area * aspect_ratio)))
            w = int(round(np.sqrt(target_area / aspect_ratio)))

            if h < img_height and w < img_width:
                x1 = np.random.randint(0, img_width - w)
                y1 = np.random.randint(0, img_height - h)

                images[i, :, y1 : y1 + h, x1 : x1 + w] = torch.rand(num_channels, h, w)

        return images, labels

    def calculate_apcer_bpcer(self, pred, label, threshold=0.5):
        # 将预测结果转换为二进制判断（大于阈值为1，小于等于阈值为0）
        binary_pred = (pred > threshold).long()

        # 计算APCER和BPCER
        num_attack_samples = torch.sum(label == 1)  # 攻击样本的总数
        num_bona_fide_samples = torch.sum(label == 0)  # 真实样本的总数

        apcer = torch.sum((binary_pred == 1) & (label == 0)) / num_bona_fide_samples
        bpcer = torch.sum((binary_pred == 0) & (label == 1)) / num_attack_samples

        return apcer, bpcer
