from torch import nn
from torch import optim
import lightning as L

from .metrics import class_separation


class X_CLIP(nn.Module):
    def __init__(self, model, preprocess):
        super(X_CLIP, self).__init__()

        self.model = model
        self.preprocess = preprocess

    def forward(self, X):
        return self.model(self.preprocess(X))


class Lit_X_CLIP(L.LightningModule):
    def __init__(self, model, metric, lr=1e-3, l2=0.01, k=None):
        super().__init__()
        self.model = model
        self.metric = metric
        self.k = k
        self.lr = lr
        self.l2 = l2

        self.save_hyperparameters(ignore=["model"])

    def training_step(self, batch):
        x, z = batch
        z_hat = self.model(x)

        if self.metric.__name__ == "mutual_knn":
            loss = -self.metric(z, z, k=self.k)
        else:
            loss = -self.metric(z_hat, z)

        self.log("train_loss", loss)
        return loss

    def validation_class_separation(self, batch):
        x, z = batch

        z_hat = self.model(x)

        r2 = class_separation(z_hat, z)
        self.log("class_separation", r2)

    def validation_alignment(self, batch):
        x, z = batch
        z_hat = self.model(x)

        if self.metric.__name__ == "mutual_knn":
            loss = -self.metric(z, z, k=self.k)
        else:
            loss = -self.metric(z_hat, z)

        self.log("validation_loss", loss)

    def validation_step(self, batch, dataloader_idx):
        if dataloader_idx == 0:
            self.validation_alignment(batch)
        elif dataloader_idx == 1:
            self.validation_class_separation(batch)

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.lr, weight_decay=self.l2)
        return optimizer
