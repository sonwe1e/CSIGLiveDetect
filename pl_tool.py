import lightning.pytorch as pl
from dataset import *
import torch
import torch.nn as nn
from lion_pytorch import Lion
from dataset import *
import torch.nn.functional as F

torch.set_float32_matmul_precision("high")

import numpy as np


def calculate_apcer_bpcer(pred, label, threshold=0.5):
    # 将预测结果转换为二进制判断（大于阈值为1，小于等于阈值为0）
    binary_pred = (pred > threshold).long()

    # 计算APCER和BPCER
    num_attack_samples = torch.sum(label == 1)  # 攻击样本的总数
    num_bona_fide_samples = torch.sum(label == 0)  # 真实样本的总数

    apcer = torch.sum((binary_pred == 1) & (label == 0)) / num_bona_fide_samples
    bpcer = torch.sum((binary_pred == 0) & (label == 1)) / num_attack_samples

    return apcer, bpcer


class TeethSegment(pl.LightningModule):
    def __init__(self, config, model):
        super().__init__()
        self.config = config
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
        self.scheduler = torch.optim.lr_scheduler.CyclicLR(
            self.optimizer,
            base_lr=4e-5,  # self.learning_rate / 2,
            max_lr=4e-4,  # self.learning_rate * 10,
            step_size_up=3890,
            mode="triangular",
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
        logits = self.model(x)
        loss = self.bce_loss(logits, y)
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
        apcer, bpcer = calculate_apcer_bpcer(self.train_predict, self.train_label)
        acer = (apcer + bpcer) / 2
        self.log("train_apcer", apcer, on_epoch=True)
        self.log("train_bpcer", bpcer, on_epoch=True)
        self.log("train_acer", acer, on_epoch=True, prog_bar=True)
        self.train_predict = []
        self.train_label = []

    def on_validation_epoch_end(self):
        self.val_predict = torch.cat(self.val_predict, dim=0)
        self.val_label = torch.cat(self.val_label, dim=0)
        apcer, bpcer = calculate_apcer_bpcer(self.val_predict, self.val_label)
        acer = (apcer + bpcer) / 2
        self.log("valid_apcer", apcer, on_epoch=True)
        self.log("valid_bpcer", bpcer, on_epoch=True)
        self.log("valid_acer", acer, on_epoch=True, prog_bar=True)
        self.val_predict = []
        self.val_label = []
