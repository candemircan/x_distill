import torch
from torch import nn
from torch import optim
import clip
import lightning as L


def load_clip(
    model_name: str, device: str
) -> tuple[torch._script.RecursiveScriptModule, torch.Compose]:
    """
    Load and return the vision encoder of the specified CLIP model and the corresponding preprocessing transforms.
    All but the final projection parameters are frozen. Only works for ViT models for now.

    Args:
        model_name (str): Must be one of OpenAI clip models. Call `clip.available_models()` to see options.
        device (str): "cpu" or "cuda"

    Returns:
        tuple[torch._script.RecursiveScriptModule, torch.Compose]: Jitted vision encoder and the image transforms.
    """
    model, preprocess = clip.load(model_name, device=device, jit=True)
    model = model.visual

    # freeze all
    for _, param in model.named_parameters():
        param.requires_grad = False

    # resnets don't have a projection layer
    if "ViT" in model_name:
        # unfreeze the final projection
        model.proj.requires_grad = True

    model.train()
    return model, preprocess


class X_CLIP(nn.Module):
    def __init__(self, model, preprocess):
        super(X_CLIP, self).__init__()

        self.model = model
        self.preprocess = preprocess

    def forward(self, X):
        return self.model(self.preprocess(X))


class Lit_X_CLIP(L.LightningModule):
    def __init__(self, model, metric, lr=1e-3, k=None):
        super().__init__()
        self.model = model
        self.metric = metric
        self.k = k
        self.lr = lr

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)

        loss = -self.metric(y_hat, y, k=self.k)

        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
