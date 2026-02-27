import torch
from lightning import Trainer

from data_module import FTIRDataModule
from lightning_model import FTIRLightningModel
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

if __name__ == "__main__": 
    data = torch.randn(200, 1, 4000)
    labels = torch.randn(200)
    batch_size = 32

    dm = FTIRDataModule(data, labels, batch_size)
    model = FTIRLightningModel(in_channels=1, latent_dim=100)

    checkpoint_cb = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1)
    earlystop_cb = EarlyStopping(monitor="val_loss", patience=3, mode="min")

    trainer = Trainer(max_epochs=10, accelerator='auto', callbacks=[checkpoint_cb, earlystop_cb])
    trainer.fit(model, dm)