import torch
from torch import nn
from pytorch_lightning.core.module import LightningModule
from torch.nn import functional as F


class AdniModel(LightningModule):
    '''
    3Dconv Model Class including the training, validation and testing steps
    '''

    def __init__(self):

        super().__init__()

        self.layer_1 = nn.Conv3d(in_channels=1, out_channels=3, kernel_size=(
            3, 3, 3), stride=(1, 1, 1))  # input (256, 256, 170) output (3, 254, 254, 168)

        self.pool1 = nn.MaxPool3d(kernel_size=(
            2, 2, 2), stride=(2, 2, 2))  # input (3, 254, 254, 168) output (3, 127, 127, 84)

        self.layer_2 = nn.Conv3d(in_channels=3, out_channels=6, kernel_size=(
            3, 3, 3), stride=(1, 1, 1))  # input (3, 127, 127, 84) output (6, 125, 125, 82)

        self.pool2 = nn.MaxPool3d(kernel_size=(
            2, 2, 2), stride=(2, 2, 2), ceil_mode=True)  # input (6, 125, 125, 82) output (6, 63, 63, 41)

        self.layer_3 = nn.Conv3d(in_channels=6, out_channels=12, kernel_size=(
            3, 3, 3), stride=(1, 1, 1))  # input (6, 63, 63, 41) output (12, 61, 61, 39)

        self.pool3 = nn.MaxPool3d(kernel_size=(
            2, 2, 2), stride=(2, 2, 2), ceil_mode=True)  # input (12, 61, 61, 39) output (12, 31, 31, 20)

        # # the input dimensions are (Number of dimensions * height * width)
        self.layer_4 = nn.Linear(12 * 31 * 31 * 20, 1000)  # 230640

        self.layer_5 = nn.Linear(1000, 100)

        self.layer_6 = nn.Linear(100, 1)

    def forward(self, x):
        """

        x is the input data

        """
        x = torch.unsqueeze(x, 0)

        x = self.pool1(F.relu(self.layer_1(x)))
        x = self.pool2(F.relu(self.layer_2(x)))
        x = self.pool3(F.relu(self.layer_3(x)))

        x = x.view(x.size(0), -1)
        print('flattened size: ', x.size())

        x = F.relu(self.layer_4(x))

        x = F.relu(self.layer_5(x))

        out = self.layer_6(x)

        out = torch.squeeze(out)

        return out

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

        return optimizer

    def training_step(self, batch, batch_idx):

        x, y = batch
        y = y.to(torch.float32)

        y_pred = self(x)

        loss = F.binary_cross_entropy(torch.sigmoid(y_pred), y.squeeze())

        # Log loss
        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):

        x, y = batch
        y = y.to(torch.float32)

        y_pred = self(x)

        loss = F.binary_cross_entropy(torch.sigmoid(y_pred), y.squeeze())

        # Log loss
        self.log('val_loss', loss)

        return loss

    def test_step(self, batch, batch_idx):

        x, y = batch
        y = y.to(torch.float32)

        y_pred = self(x)

        loss = F.binary_cross_entropy(torch.sigmoid(y_pred), y.squeeze())

        self.log("loss", loss)

        return loss
