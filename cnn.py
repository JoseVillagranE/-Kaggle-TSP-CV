import pytorch_lightning as pl
from torch.optim import Adam
from cnn_models import AlexNet
import torch
from torch.nn import MSELoss


class cnn(pl.LightningModule):
    def __init__(self, lr=1e-3, eps=1e-08, wd=1e-3):

        super().__init__()
        self.lr = lr
        self.eps = eps
        self.wd = wd
        self.model = AlexNet()
        self.mse = MSELoss()

    def forward(self, input):
        outps = self.model(input)
        return outps

    def training_step(self, batch, batch_idx):

        imgs, labels = batch
        outp = self.forward(imgs)
        loss = self.mse(outp.squeeze(), labels.float())
        return loss

    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx)

    def training_epoch_end(self, outputs):

        outputs = [o["loss"] for o in outputs]
        avg_loss = torch.stack(outputs).mean()
        self.log("avg_training_loss", avg_loss)

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack(outputs).mean()
        self.log("avg_validation_loss", avg_loss)

    def configure_optimizers(self):

        optimizer = Adam(
            [p for p in self.parameters() if p.requires_grad],
            lr=self.lr,
            eps=self.eps,
            weight_decay=self.wd,
        )

        return [optimizer]

    def save_pretrained(self, path):
        pass

    def load_pretrained(self, path):
        pass
