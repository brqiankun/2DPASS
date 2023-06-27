import os
from typing import Any
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import pytorch_lightning as pl

encoder = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))
decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))

class LitAutoEncoder(pl.LightningModule):
    def __init__(self, encoder, decoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        # defines the train loop
        # independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        # logging to tensorboard(if installed) by default
        self.log("train_loss", loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

def test():
    # init the autoencoder
    autoencoder = LitAutoEncoder(encoder, decoder)

    dataset = MNIST(os.getcwd(), download=True, transform=ToTensor())
    train_loader = utils.data.DataLoader(dataset)

    # train the model
    trainer = pl.Trainer(max_epochs=1, limit_train_batches=100)
    trainer.fit(model=autoencoder, train_dataloader=train_loader)

    # raise RuntimeError("stop here")
    # use the model
    checkpoint = "./lightning_logs/version_0/checkpoints/epoch=0-step=99.ckpt"
    autoencoder = LitAutoEncoder.load_from_checkpoint(checkpoint, encoder=encoder, decoder=decoder)

    # choose your trained nn.Moudle
    encoder = autoencoder.encoder
    encoder.eval()

    # embed 4 fake images
    fake_image_batch = Tensor(4, 28 * 28)
    embeddings = encoder(fake_image_batch)
    print("⚡" * 20, "\nPredictions (4 image embeddings):\n", embeddings, "\n", "⚡" * 20)






