from asyncio import base_tasks
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from torchvision.transforms import Resize, Compose, ToTensor

from tsp_cv_dataset import TSPCV_Dataset
from cnn import cnn


def main(cfg):

    transform = Compose([Resize(cfg["size"]), ToTensor()])

    train_dataset = TSPCV_Dataset(cfg["train_path"], transform)
    val_dataset = TSPCV_Dataset(cfg["test_path"], transform)

    train_dataloader = DataLoader(train_dataset, batch_size=cfg["batch_size"])
    test_dataloader = DataLoader(val_dataset, batch_size=cfg["batch_size"])

    model = cnn(lr=cfg["lr"], eps=cfg["eps"], wd=cfg["wd"])

    checkpoint_callback = ModelCheckpoint(
        dirpath="./Weights", filename="checkpoint"
    )
    wandb_logger = WandbLogger(name='TSP-CV_training', project='kaggle_project')


    trainer = pl.Trainer(
        gpus=cfg["gpus"],
        max_epochs=cfg["max_epochs"],
        check_val_every_n_epoch=cfg["val_every_n"],
        callbacks=[checkpoint_callback],
        logger= wandb_logger
    )

    trainer.fit(model, train_dataloader)

    trainer.test(dataloaders=test_dataloader)


if __name__ == "__main__":

    cfg = {
        "batch_size": 64,
        "eps": 1e-8,
        "gpus": 1 if torch.cuda.is_available() else 0,
        "lr": 1e-3,
        "max_epochs": 10,
        "size": (329, 200),
        "test_path": "data/test.csv",
        "train_path": "data/train.csv",
        "val_every_n": 2,
        "wd": 1e-5,
    }

    main(cfg)
