import lightning
import torch 
from model import FTIREncoder
import torch.nn as nn 

class FTIRLightningModel(lightning.LightningModule):
    def __init__(self, in_channels, latent_dim):
        super().__init__()
        self.model = FTIREncoder(in_channels, latent_dim)
        self.loss_fn = nn.MSELoss()

    def training_step(self, batch, batch_idx):
        x, y = batch 
        yhat = self.model(x)
        loss = self.loss_fn(yhat, y)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch 
        yhat = self.model(x)
        loss = self.loss_fn(yhat, y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

if __name__ == "__main__":
    print(FTIRLightningModel(in_channels=1, latent_dim=100))