from asyncio import base_tasks
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.transforms import Resize, Compose, ToTensor

from tsp_cv_dataset import TSPCV_Dataset
from cnn import cnn

def training(cfg):

    transform = Compose([Resize(cfg["size"]), ToTensor()])

    dataset = TSPCV_Dataset(cfg["train_path"], transform)
    dataset_size = len(dataset)

    indices = list(range(dataset_size))
    split = int(np.floor(cfg["validation_split"] * dataset_size))
    
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg["batch_size"], 
                                               sampler=train_sampler)
    validation_dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg["batch_size"],
                                                    sampler=valid_sampler)

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

    trainer.fit(model, train_dataloader, validation_dataloader)


# def testing(cfg):

#     trainer.test(dataloaders=test_dataloader)

if __name__ == "__main__":

    cfg = {
        "batch_size": 64,
        "eps": 1e-8,
        "gpus": 1 if torch.cuda.is_available() else 0,
        "lr": 1e-3,
        "max_epochs": 10,
        "size": (329, 200),
        "train_path": "data/train.csv",
        "val_every_n": 2,
        "validation_split": 0.2,
        "wd": 1e-5,
    }

    main(cfg)
