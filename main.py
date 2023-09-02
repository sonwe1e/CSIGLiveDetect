import argparse
import timm
from dataset import *
from pl_tool import TeethSegment
import torch
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger

torch.set_float32_matmul_precision("high")


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-b", "--batch_size", type=int, default=512)
    parser.add_argument("-l", "--learning_rate", type=float, default=6e-4)
    parser.add_argument("-wd", "--weight_decay", type=float, default=2e-4)
    parser.add_argument("-p", "--precision", type=str, default="bf16-mixed")
    parser.add_argument("-e", "--epochs", type=int, default=20)
    parser.add_argument("--exp_name", type=str, default="baseline")
    parser.add_argument("-f", "--fold", type=int, default=5)
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-dp", "--dropout", type=float, default=0.5)
    return parser.parse_args()


if __name__ == "__main__":
    config = get_args()
    pl.seed_everything(config.seed)
    train_dataloader = get_dataloader(
        bs=config.batch_size,
        split="train",
        transform=train_transform,
        fold=config.fold,
    )
    valid_dataloader = get_dataloader(
        bs=config.batch_size,
        split="valid",
        transform=valid_transform,
        fold=config.fold,
    )
    # wandb.save("*.py")
    wandb_logger = WandbLogger(
        project="LiveDetect", name=config.exp_name, offline=0, config=config
    )
    model = timm.create_model("resnet34", num_classes=1)
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(config.dropout), torch.nn.Linear(512, 1)
    )
    # from model import ResNet

    # model = ResNet()
    # model = torch.compile(model)
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[config.device],
        max_epochs=config.epochs,
        precision=config.precision,
        default_root_dir="./",
        logger=wandb_logger,
        log_every_n_steps=5,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath=f"./LiveDetect/{config.exp_name}/",
                monitor="valid_acer",
                mode="min",
                save_top_k=1,
                save_last=False,
                filename="{epoch}-{valid_acer:.4f}",
            ),
        ],
    )
    trainer.fit(
        TeethSegment(config, model, len(train_dataloader)),
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
    )
