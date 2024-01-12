import torch
from torch import nn
from pytorch_lightning.core.module import LightningModule
from torch.nn import functional as F
from monai.networks.nets import resnet10
torch.backends.cudnn.enabled = False
import numpy as np

class SiameseResNetModel(LightningModule):
    def __init__(self, class_weights):
        super().__init__()
        self.class_weights = class_weights

        # Create two instances of resnet10
        self.net1 = resnet10(pretrained=False, spatial_dims=3, n_input_channels=1, num_classes=5)
        self.net2 = resnet10(pretrained=False, spatial_dims=3, n_input_channels=1, num_classes=5)

        # You may need to fine-tune the output dimensions based on your use case
        # self.fc = nn.Linear(400, 5)

        self.loss = ContrastiveLoss()  # You need to define ContrastiveLoss

    def forward(self, x1, x2):
        out1 = self.net1(x1)
        out2 = self.net2(x2)
        return out1, out2

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, capturable=True)
        return optimizer

    def calculate_balanced_accuracy(self, y_pred, y_true, class_weights):
        accuracy_per_class = (y_pred == y_true).float()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if device.type == "cuda":
            weighted_accuracy_per_class = accuracy_per_class * torch.Tensor([class_weights[label] for label in y_true]).cuda()
        else:
            accuracy_per_class = accuracy_per_class.cpu()
            weighted_accuracy_per_class = accuracy_per_class * torch.Tensor([class_weights[label] for label in y_true]).cpu()

        balanced_acc = weighted_accuracy_per_class.sum() / len(y_true)

        return balanced_acc

    def training_step(self, batch, batch_idx):
        x1, x2, y = batch
        y = y.to(torch.float32)

        # Forward pass through the Siamese network
        out1, out2 = self(x1, x2)

        # Calculate the contrastive loss
        loss = self.loss(out1, out2, y)

        # Calculate accuracy and balanced accuracy
        y_pred = (F.cosine_similarity(out1, out2) > 0).float()
        acc = (y_pred == y).float().mean()
        balanced_acc = self.calculate_balanced_accuracy(y_pred, y, self.class_weights)

        # Log metrics
        self.log('train_loss', loss)
        self.log('train_acc', acc, prog_bar=True)
        self.log('train_balanced_acc', balanced_acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x1, x2, y = batch
        y = y.to(torch.float32)

        # Forward pass through the Siamese network
        out1, out2 = self(x1, x2)

        # Calculate the contrastive loss
        loss = self.loss(out1, out2, y)

        # Calculate accuracy and balanced accuracy
        y_pred = (F.cosine_similarity(out1, out2) > 0).float()
        acc = (y_pred == y).float().mean()
        balanced_acc = self.calculate_balanced_accuracy(y_pred, y, self.class_weights)

        # Log metrics
        self.log('val_loss', loss)
        self.log('val_acc', acc, prog_bar=True)
        self.log('val_balanced_acc', balanced_acc, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        x1, x2, y = batch
        y = y.to(torch.float32)

        # Forward pass through the Siamese network
        out1, out2 = self(x1, x2)

        # Calculate the contrastive loss
        loss = self.loss(out1, out2, y)

        # Calculate accuracy and balanced accuracy
        y_pred = (F.cosine_similarity(out1, out2) > 0).float()
        acc = (y_pred == y).float().mean()
        balanced_acc = self.calculate_balanced_accuracy(y_pred, y, self.class_weights)

        # Log metrics
        self.log('test_loss', loss)
        self.log('test_acc', acc, prog_bar=True)
        self.log('test_balanced_acc', balanced_acc, prog_bar=True)

        return loss

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, out1, out2, y):
        similarity = self.cosine_similarity(out1, out2)
        loss = 0.5 * ((1 - y) * similarity.pow(2) +
                      y * F.relu(self.margin - similarity).pow(2))
        return loss.mean()
