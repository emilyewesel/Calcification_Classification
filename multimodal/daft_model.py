import torch
from torch import nn
from pytorch_lightning.core.module import LightningModule
from torch.nn import functional as F
from monai.networks.nets import resnet10, resnet18, resnet34, resnet50
from settings import IMAGE_SIZE, NUM_FEATURES, BATCH_SIZE
import torchmetrics
import pandas as pd
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from typing import Any, Dict, Optional, Sequence
from torch.nn import Module

class DAFTModel(LightningModule):
    '''
    DAFT ResNet Model Class including the training, validation and testing steps
    '''

    def __init__(self, class_weight, scaler):

        super().__init__()

        self.NUM_FEATURES = NUM_FEATURES
        self.class_weight = class_weight
        self.scaler = scaler

        in_channels=  1
        n_outputs = 1
        bn_momentum = 0.1
        n_basefilters = 4
        filmblock_args: Optional[Dict[Any, Any]] = None

        if filmblock_args is None:
            filmblock_args = {}

        self.split_size = 4 * n_basefilters
        self.conv1 = ConvBnReLU(in_channels, n_basefilters, bn_momentum=bn_momentum)
        self.pool1 = nn.MaxPool3d(2, stride=2)  # 32
        self.block1 = ResBlock(n_basefilters, n_basefilters, bn_momentum=bn_momentum)
        self.block2 = ResBlock(n_basefilters, 2 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 16
        self.block3 = ResBlock(2 * n_basefilters, 4 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 8
        self.blockX = DAFTBlock(4 * n_basefilters, 8 * n_basefilters, bn_momentum=bn_momentum, **filmblock_args)  # 4
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(8 * n_basefilters, n_outputs)

        self.loss_func = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.class_weight).float())

        self.train_macro_accuracy = torchmetrics.Accuracy(task='multiclass', average='macro', num_classes=2)

        self.val_macro_accuracy = torchmetrics.Accuracy(task='multiclass', average='macro', num_classes=2)

        self.test_macro_accuracy = torchmetrics.Accuracy(task='multiclass', average='macro', num_classes=2)

        self.train_accuracy = torchmetrics.Accuracy(task='multiclass', average='micro', num_classes=2)

        self.val_accuracy = torchmetrics.Accuracy(task='multiclass', average='micro', num_classes=2)

        self.test_accuracy = torchmetrics.Accuracy(task='multiclass', average='micro', num_classes=2)

        self.train_macro_f1 = torchmetrics.classification.MulticlassF1Score(task='multiclass', num_classes=2, average='macro')

        self.train_auc = torchmetrics.classification.BinaryAUROC(task='multiclass', num_classes=2, average='macro')

        self.val_macro_f1 = torchmetrics.classification.MulticlassF1Score(task='multiclass', num_classes=2, average='macro')

        self.val_auc = torchmetrics.classification.BinaryAUROC(task='multiclass', num_classes=2, average='macro')

        self.test_macro_f1 = torchmetrics.classification.MulticlassF1Score(task='multiclass', num_classes=2, average='macro')
        
        self.test_auc = torchmetrics.classification.BinaryAUROC(task='multiclass', num_classes=2, average='macro')

        self.results_column_names = ['subject', 'label', 'prediction', 'age', 'sex']

        self.train_results_df = pd.DataFrame(columns=self.results_column_names)

        self.train_results_df_all = pd.DataFrame(columns=self.results_column_names)

        self.val_results_df_all = pd.DataFrame(columns=self.results_column_names)

        self.val_results_df = pd.DataFrame(columns=self.results_column_names)

    def forward(self, img, tab):
        """

        x is the input data

        """
        # run the model for the image
        img = torch.unsqueeze(img, 1)
        tab = tab.to(torch.float32)

        out = self.conv1(img)
        out = self.pool1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.blockX(out, tab)
        out = self.global_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        out = torch.squeeze(out)

        return out

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[18,27], gamma=0.1)

        lr_scheduler = {
            'scheduler': scheduler,
            'name': 'lr_logging'
        }

        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        print("daft train")
        img, tab, y, subject_id = batch

        y = y.to(torch.float)
        img = img.type(torch.float)
        y_pred = self(img, tab)

        loss = self.loss_func(y_pred, y.squeeze())

        y_pred_tag = torch.round(torch.sigmoid(y_pred))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #if device.type == "cpu":
        self.train_results_df['subject'] = tuple(subject_id)
        self.train_results_df['label'] = y.squeeze().detach().cpu().numpy()
        self.train_results_df['prediction'] = y_pred_tag.detach().cpu().numpy()
        tab_bef_normalization = self.scaler.inverse_transform(tab.detach().cpu().numpy())
        # else:
        #     self.train_results_df['subject'] = tuple(subject_id)
        #     self.train_results_df['label'] = y.squeeze().detach().cuda().numpy()
        #     self.train_results_df['prediction'] = y_pred_tag.detach().cuda().numpy()
        #     tab_bef_normalization = self.scaler.inverse_transform(tab.detach().cuda().numpy())
        self.train_results_df['age'] = tab_bef_normalization[:,2]
        self.train_results_df['sex'] = tab_bef_normalization[:, 1]

        self.train_results_df_all = pd.concat([self.train_results_df_all , self.train_results_df], ignore_index=True)


        # loss = F.binary_cross_entropy(torch.sigmoid(y_pred), y.squeeze())
        if BATCH_SIZE == 1:
            self.train_accuracy(torch.unsqueeze(y_pred_tag, 0), y)

            self.train_macro_accuracy(torch.unsqueeze(y_pred_tag, 0), y)
            self.train_auc(torch.unsqueeze(y_pred_tag, 0), y)
            self.train_macro_f1(torch.unsqueeze(y_pred_tag, 0), y)

        else:
            self.train_accuracy(y_pred_tag, y)

            self.train_macro_accuracy(y_pred_tag, y)
            self.train_auc(y_pred_tag, y)
            self.train_macro_f1(y_pred_tag, y)
        self.log('train_acc_step', self.train_accuracy, on_step=False, on_epoch=True)
        self.log('train_macro_acc_step', self.train_macro_accuracy, on_step=True, on_epoch=True)
        self.log('train_auc', self.train_auc, on_step=False, on_epoch=True)
        self.log('train_macro_f1', self.train_macro_f1, on_step=False, on_epoch=True)
        # Log loss
        self.log('train_loss', loss, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):

        img, tab, y, subject_id = batch

        y = y.to(torch.float)
        img = img.type(torch.float)
        y_pred = self(img, tab)

        loss = self.loss_func(y_pred, y.squeeze())

        y_pred_tag = torch.round(torch.sigmoid(y_pred))

        self.val_results_df['subject'] = tuple(subject_id)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #if device.type == "cpu":
        self.val_results_df['label'] = y.squeeze().detach().cpu().numpy()
        self.val_results_df['prediction'] = y_pred_tag.detach().cpu().numpy()
        tab_bef_normalization = self.scaler.inverse_transform(tab.detach().cpu().numpy())
        # else: 
        #     self.val_results_df['label'] = y.squeeze().detach().cuda().numpy()
        #     self.val_results_df['prediction'] = y_pred_tag.detach().cuda().numpy()
        #     tab_bef_normalization = self.scaler.inverse_transform(tab.detach().cuda().numpy())
        self.val_results_df['age'] = tab_bef_normalization[:,2]
        self.val_results_df['sex'] = tab_bef_normalization[:, 1]


        if BATCH_SIZE == 1:

            self.val_accuracy(torch.unsqueeze(y_pred_tag, 0), y)
            self.val_macro_accuracy(torch.unsqueeze(y_pred_tag, 0), y)
            # self.val_auc(torch.unsqueeze(y_pred_tag, 0), y)
            # self.val_macro_f1(torch.unsqueeze(y_pred_tag, 0), y)

        else:
            self.val_accuracy(y_pred_tag, y)
            self.val_macro_accuracy(y_pred_tag, y)
            # self.val_auc(y_pred_tag, y)
            # self.val_macro_f1(y_pred_tag, y)
        self.log('val_acc_step', self.val_accuracy, on_step=False, on_epoch=True)
        self.log('val_macro_acc_step', self.val_macro_accuracy, on_step=True, on_epoch=True, prog_bar=True)
        # self.log('val_auc', self.val_auc, on_step=False, on_epoch=True)
        # self.log('val_macro_f1', self.val_macro_f1, on_step=False, on_epoch=True)

        # Log loss
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):

        img, tab, y = batch
        y = y.to(torch.float32)

        y_pred = self(img, tab)

        loss = self.loss_func(y_pred, y.squeeze())  # loss = F.binary_cross_entropy(torch.sigmoid(y_pred), y.squeeze(), pos_weights = )

        y_pred_tag = torch.round(torch.sigmoid(y_pred))

        if BATCH_SIZE == 1:
            self.test_accuracy(torch.unsqueeze(y_pred_tag, 0), y)
            self.test_macro_accuracy(torch.unsqueeze(y_pred_tag, 0), y)
            #  self.test_auc(torch.unsqueeze(y_pred_tag, 0), y)
            self.test_macro_f1(torch.unsqueeze(y_pred_tag, 0), y)

        else:
            self.test_accuracy(y_pred_tag, y)
            self.test_macro_accuracy(y_pred_tag, y)
            # self.test_auc(y_pred_tag, y)
            self.test_macro_f1(y_pred_tag, y)
        self.log('test_acc_step', self.test_accuracy, on_step=True, on_epoch=False)
        self.log('test_macro_acc_step', self.test_macro_accuracy, on_step=True, on_epoch=True)
        self.log("test loss", loss)
        # self.log('test_auc', self.test_auc, on_step=True, on_epoch=True)
        self.log('test_macro_f1', self.test_macro_f1, on_step=True, on_epoch=True)
        

        return loss

    def training_epoch_end(self, outs):
        # log epoch metric

        filename_out = 'results/train_out_daft_' + str(self.current_epoch) + '_' + self.trainer.logger.experiment.name + '.csv'
        self.train_results_df_all.to_csv(filename_out)

        # Clear the dataframe so the new epoch can start fresh
        self.train_results_df_all = pd.DataFrame(columns=self.results_column_names)

        self.log('train_acc_epoch', self.train_accuracy)
        self.log('train_macro_acc_epoch', self.train_macro_accuracy)
        self.log('train_f1', self.train_macro_f1)
        # self.log('train_auc', self.train_auc)


    def validation_epoch_end(self, outputs):
        # log epoch metric

        filename_out = 'results/val_out_daft_' + str(self.current_epoch) + '_' + self.trainer.logger.experiment.name + '.csv'
        self.val_results_df_all.to_csv(filename_out)

        # Clear the dataframe so the new epoch can start fresh
        self.val_results_df_all = pd.DataFrame(columns=self.results_column_names)

        self.log('val_acc_epoch', self.val_accuracy)
        self.log('val_macro_acc_epoch', self.val_macro_accuracy)
        self.log('val_f1', self.val_macro_f1)
        # self.log('val_auc', self.val_auc)


def conv3d(in_channels, out_channels, kernel_size=3, stride=1):
    if kernel_size != 1:
        padding = 1
    else:
        padding = 0
    return nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)


class ConvBnReLU(nn.Module):
    def __init__(
        self, in_channels, out_channels, bn_momentum=0.05, kernel_size=3, stride=1, padding=1,
    ):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm3d(out_channels, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn_momentum=0.05, stride=1):
        super().__init__()
        self.conv1 = conv3d(in_channels, out_channels, stride=stride)
        self.bn1 = nn.BatchNorm3d(out_channels, momentum=bn_momentum)
        self.conv2 = conv3d(out_channels, out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm3d(out_channels, momentum=bn_momentum),
            )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)

        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class FilmBase(nn.Module, metaclass=ABCMeta):
    """Absract base class for models that are related to FiLM of Perez et al"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bn_momentum: float,
        stride: int,
        ndim_non_img: int,
        location: int,
        activation: str,
        scale: bool,
        shift: bool,
    ) -> None:

        super().__init__()

        # sanity checks
        if location not in set(range(5)):
            raise ValueError(f"Invalid location specified: {location}")
        if activation not in {"tanh", "sigmoid", "linear"}:
            raise ValueError(f"Invalid location specified: {location}")
        if (not isinstance(scale, bool) or not isinstance(shift, bool)) or (not scale and not shift):
            raise ValueError(
                f"scale and shift must be of type bool:\n    -> scale value: {scale}, "
                "scale type {type(scale)}\n    -> shift value: {shift}, shift type: {type(shift)}"
            )
        # ResBlock
        self.conv1 = conv3d(in_channels, out_channels, stride=stride)
        self.bn1 = nn.BatchNorm3d(out_channels, momentum=bn_momentum, affine=(location != 3))
        self.conv2 = conv3d(out_channels, out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm3d(out_channels, momentum=bn_momentum),
            )
        else:
            self.downsample = None
        # Film-specific variables
        self.location = location
        if self.location == 2 and self.downsample is None:
            raise ValueError("This is equivalent to location=1 and no downsampling!")
        # location decoding
        self.film_dims = 0
        if location in {0, 1, 2}:
            self.film_dims = in_channels
        elif location in {3, 4}:
            self.film_dims = out_channels
        if activation == "sigmoid":
            self.scale_activation = nn.Sigmoid()
        elif activation == "tanh":
            self.scale_activation = nn.Tanh()
        elif activation == "linear":
            self.scale_activation = None

    @abstractmethod
    def rescale_features(self, feature_map, x_aux):
        """method to recalibrate feature map x"""

    def forward(self, feature_map, x_aux):

        if self.location == 0:
            feature_map = self.rescale_features(feature_map, x_aux)
        residual = feature_map

        if self.location == 1:
            residual = self.rescale_features(residual, x_aux)

        if self.location == 2:
            feature_map = self.rescale_features(feature_map, x_aux)
        out = self.conv1(feature_map)
        out = self.bn1(out)

        if self.location == 3:
            out = self.rescale_features(out, x_aux)
        out = self.relu(out)

        if self.location == 4:
            out = self.rescale_features(out, x_aux)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(residual)
        out += residual
        out = self.relu(out)

        return out


class FilmBlock(FilmBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bn_momentum: float = 0.1,
        stride: int = 2,
        ndim_non_img: int = NUM_FEATURES,
        location: int = 0,
        activation: str = "linear",
        scale: bool = True,
        shift: bool = True,
        bottleneck_dim: int = 32,
    ):

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            bn_momentum=bn_momentum,
            stride=stride,
            ndim_non_img=ndim_non_img,
            location=location,
            activation=activation,
            scale=scale,
            shift=shift,
        )

        self.bottleneck_dim = bottleneck_dim
        # shift and scale decoding
        self.split_size = 0
        if scale and shift:
            self.split_size = self.film_dims
            self.scale = None
            self.shift = None
            self.film_dims = 2 * self.film_dims
        elif not scale:
            self.scale = 1
            self.shift = None
        elif not shift:
            self.shift = 0
            self.scale = None
        # create aux net
        layers = [
            ("aux_base", nn.Linear(ndim_non_img, self.bottleneck_dim, bias=False)),
            ("aux_relu", nn.ReLU()),
            ("aux_out", nn.Linear(self.bottleneck_dim, self.film_dims, bias=False)),
        ]
        self.aux = nn.Sequential(OrderedDict(layers))

    def rescale_features(self, feature_map, x_aux):

        attention = self.aux(x_aux)

        assert (attention.size(0) == feature_map.size(0)) and (
            attention.dim() == 2
        ), f"Invalid size of output tensor of auxiliary network: {attention.size()}"

        if self.scale == self.shift:
            v_scale, v_shift = torch.split(attention, self.split_size, dim=1)
            v_scale = v_scale.view(*v_scale.size(), 1, 1, 1).expand_as(feature_map)
            v_shift = v_shift.view(*v_shift.size(), 1, 1, 1).expand_as(feature_map)
            if self.scale_activation is not None:
                v_scale = self.scale_activation(v_scale)
        elif self.scale is None:
            v_scale = attention
            v_scale = v_scale.view(*v_scale.size(), 1, 1, 1).expand_as(feature_map)
            v_shift = self.shift
            if self.scale_activation is not None:
                v_scale = self.scale_activation(v_scale)
        elif self.shift is None:
            v_scale = self.scale
            v_shift = attention
            v_shift = v_shift.view(*v_shift.size(), 1, 1, 1).expand_as(feature_map)
        else:
            raise AssertionError(
                f"Sanity checking on scale and shift failed. Must be of type bool or None: {self.scale}, {self.shift}"
            )

        return (v_scale * feature_map) + v_shift


class DAFTBlock(FilmBase):
    # Block for ZeCatNet
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bn_momentum: float = 0.1,
        stride: int = 2,
        ndim_non_img: int = NUM_FEATURES,
        location: int = 0,
        activation: str = "linear",
        scale: bool = True,
        shift: bool = True,
        bottleneck_dim: int = 32,
    ) -> None:

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            bn_momentum=bn_momentum,
            stride=stride,
            ndim_non_img=ndim_non_img,
            location=location,
            activation=activation,
            scale=scale,
            shift=shift,
        )

        self.bottleneck_dim = bottleneck_dim
        aux_input_dims = self.film_dims
        # shift and scale decoding
        self.split_size = 0
        if scale and shift:
            self.split_size = self.film_dims
            self.scale = None
            self.shift = None
            self.film_dims = 2 * self.film_dims
        elif not scale:
            self.scale = 1
            self.shift = None
        elif not shift:
            self.shift = 0
            self.scale = None

        # create aux net
        layers = [
            ("aux_base", nn.Linear(ndim_non_img + aux_input_dims, self.bottleneck_dim, bias=False)),
            ("aux_relu", nn.ReLU()),
            ("aux_out", nn.Linear(self.bottleneck_dim, self.film_dims, bias=False)),
        ]
        self.aux = nn.Sequential(OrderedDict(layers))

    def rescale_features(self, feature_map, x_aux):

        squeeze = self.global_pool(feature_map)
        squeeze = squeeze.view(squeeze.size(0), -1)
        squeeze = torch.cat((squeeze, x_aux), dim=1)

        attention = self.aux(squeeze)
        if self.scale == self.shift:
            v_scale, v_shift = torch.split(attention, self.split_size, dim=1)
            v_scale = v_scale.view(*v_scale.size(), 1, 1, 1).expand_as(feature_map)
            v_shift = v_shift.view(*v_shift.size(), 1, 1, 1).expand_as(feature_map)
            if self.scale_activation is not None:
                v_scale = self.scale_activation(v_scale)
        elif self.scale is None:
            v_scale = attention
            v_scale = v_scale.view(*v_scale.size(), 1, 1, 1).expand_as(feature_map)
            v_shift = self.shift
            if self.scale_activation is not None:
                v_scale = self.scale_activation(v_scale)
        elif self.shift is None:
            v_scale = self.scale
            v_shift = attention
            v_shift = v_shift.view(*v_shift.size(), 1, 1, 1).expand_as(feature_map)
        else:
            raise AssertionError(
                f"Sanity checking on scale and shift failed. Must be of type bool or None: {self.scale}, {self.shift}"
            )

        return (v_scale * feature_map) + v_shift

def check_is_unique(values: Sequence[Any]) -> bool:
    if len(values) != len(set(values)):
        raise ValueError("values of list must be unique")


class BaseModel(Module, metaclass=ABCMeta):
    """Abstract base class for models that can be executed by
    :class:`daft.training.train_and_eval.ModelRunner`.
    """

    @abstractmethod
    def __init__(self) -> None:
        super().__init__()
        check_is_unique(self.input_names)
        check_is_unique(self.output_names)
        check_is_unique(list(self.input_names) + list(self.output_names))

    @property
    @abstractmethod
    def input_names(self) -> Sequence[str]:
        """Names of parameters passed to self.forward"""

    @property
    @abstractmethod
    def output_names(self) -> Sequence[str]:
        """Names of tensors returned by self.forward"""

