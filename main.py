import argparse
import timm
from dataset import *
from pl_tool import TeethSegment
import torch
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
import wandb
torch.set_float32_matmul_precision("high")


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-b", "--batch_size", type=int, default=500)
    parser.add_argument("-l", "--learning_rate", type=float, default=4e-4)
    parser.add_argument("-wd", "--weight_decay", type=float, default=4e-4)
    parser.add_argument("-p", "--precision", type=str, default="bf16-mixed")
    parser.add_argument("-e", "--epochs", type=int, default=100)
    parser.add_argument("--exp_name", type=str, default="0")
    parser.add_argument("-f", "--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


if __name__ == "__main__":
    # torch.multiprocessing.set_start_method('spawn', force=True)
    config = get_args()
    pl.seed_everything(config.seed)
    for i in range(6):
        config.fold = i
        config.exp_name = str(i)
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
            log_every_n_steps=5,
            callbacks=[
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
            # ckpt_path='./epoch=97-valid_acer=0.0170.ckpt'
        )
        wandb.finish()
