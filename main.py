import argparse
import timm
from dataset import *
from pl_tool import TeethSegment
import torch
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger

# torch.set_float32_matmul_precision("high")


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-b", "--batch_size", type=int, default=512)
    parser.add_argument("-l", "--learning_rate", type=float, default=6e-4)
    parser.add_argument("-wd", "--weight_decay", type=float, default=1e-3)
    parser.add_argument("-p", "--precision", type=str, default="bf16-mixed")
    parser.add_argument("-e", "--epochs", type=int, default=120)
    parser.add_argument("--exp_name", type=str, default="0")
    parser.add_argument("-f", "--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


if __name__ == "__main__":
    config = get_args()
    pl.seed_everything(config.seed)
    for i in range(1, 5):
        config.fold = i
        config.exp_name = f"mixup0.3+randomcropresize0.64+fold5+{i}"
        train_dataloader = get_dataloader(
            bs=config.batch_size,
            split="train",
            transform=train_transform,
            fold=config.fold,
        )
        valid_dataloader = get_dataloader(
            bs=config.batch_size,
            split="valid",
            transform=test_transform,
            fold=config.fold,
        )
        # wandb.save("*.py")
        wandb_logger = WandbLogger(project="LiveDetect", name=config.exp_name)
        model = timm.create_model("resnet34", pretrained=False, num_classes=1)
        # model = torch.compile(model)
        trainer = pl.Trainer(
            accelerator="gpu",
            devices=[0],
            max_epochs=config.epochs,
            precision=config.precision,
            default_root_dir="./",
            deterministic=False,
            logger=wandb_logger,
            # limit_train_batches=0.2,
            callbacks=[
                # pl.callbacks.LearningRateFinder(),
                pl.callbacks.ModelCheckpoint(
                    dirpath=f"./LiveDetect/{config.exp_name}/",
                    monitor="valid_acer",
                    mode="min",
                    save_top_k=5,
                    save_last=True,
                    filename="{epoch}-{valid_acer:.4f}",
                ),
            ],
        )
        trainer.fit(
            TeethSegment(config, model),
            train_dataloaders=train_dataloader,
            val_dataloaders=valid_dataloader,
        )