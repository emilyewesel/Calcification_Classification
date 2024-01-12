import torch
from torch import nn
from pytorch_lightning.core.module import LightningModule
from torch.nn import functional as F
from monai.networks.nets import resnet10, resnet18, resnet34, resnet50
from settings import IMAGE_SIZE, FEATURES, BATCH_SIZE, TARGET
import torchmetrics
import pandas as pd


class MultiModModel(LightningModule):
    '''
    Resnet Model Class including the training, validation and testing steps
    '''

    def __init__(self, class_weight, scaler):

        super().__init__()
        
        self.class_weight = class_weight

        self.scaler = scaler

        self.resnet = resnet10(pretrained=False,
                              spatial_dims=3,
                              num_classes=120,
                              n_input_channels=1
                              )

        self.NUM_FEATURES = len(FEATURES)

        # fc layer for tabular data
        self.fc1 = nn.Linear(self.NUM_FEATURES, 120)

        # first fc layer which takes concatenated imput
        self.fc2 = nn.Linear((120 + 120), 32)
        
        # final fc layer which takes concatenated imput
        self.fc3 = nn.Linear(32, 1)
        
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

        # print(img.shape)
        img = torch.unsqueeze(img, 1)
        img = img.to(torch.float32)
        # print(img.type)
        # print(img.shape)
        
        img = self.resnet(img)

        # change the dtype of the tabular data
        tab = tab.to(torch.float32)

        # forward tabular data
        tab = F.relu(self.fc1(tab))
        
        # concat image and tabular data
        x = torch.cat((img, tab), dim=1)

        x = F.relu(self.fc2(x))

        out = self.fc3(x)

        out = torch.squeeze(out)

        return out

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20,35,50], gamma=0.8)

        lr_scheduler = {
            'scheduler': scheduler,
            'name': 'lr_logging'
        }

        return [optimizer], [lr_scheduler]
        

    def training_step(self, batch, batch_idx):
        print("train model.py")

        img, tab, y, subject_id = batch
        
        img = img.clone().detach().requires_grad_(True).float() #torch.tensor(img).float()
        
        y = y.to(torch.float32)

        y_pred = self(img, tab)

        loss_func = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.class_weight).float())

        loss = loss_func(y_pred, y.squeeze())
        
        y_pred_tag = torch.round(torch.sigmoid(y_pred))
        
        self.train_results_df['subject'] = tuple(subject_id)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #if device.type == "cpu":
        self.train_results_df['label'] = y.squeeze().detach().cpu().numpy()
        self.train_results_df['prediction'] = y_pred_tag.detach().cpu().numpy()
        tab_bef_normalization = self.scaler.inverse_transform(tab.detach().cpu().numpy())
        # else: 
        #     self.train_results_df['label'] = y.squeeze().detach().cuda().numpy()
        #     self.train_results_df['prediction'] = y_pred_tag.detach().cuda().numpy()
        #     tab_bef_normalization = self.scaler.inverse_transform(tab.detach().cuda().numpy())
        self.train_results_df['age'] = tab_bef_normalization[:,2]
        self.train_results_df['sex'] = tab_bef_normalization[:, 1]
        
        self.train_results_df_all = pd.concat([self.train_results_df_all , self.train_results_df], ignore_index=True)
        
        if BATCH_SIZE == 1:
            self.train_accuracy(torch.unsqueeze(y_pred_tag, 0), y)
            
            self.train_macro_accuracy(torch.unsqueeze(y_pred_tag, 0), y)
            self.train_macro_f1(torch.unsqueeze(y_pred_tag, 0), y)
            self.train_auc(torch.unsqueeze(y_pred_tag, 0), y)
        else:
            self.train_accuracy(y_pred_tag, y)
            
            self.train_macro_accuracy(y_pred_tag, y)
            self.train_macro_f1(y_pred_tag, y)
            self.train_auc(y_pred_tag, y)
        
        self.log('train_acc_step', self.train_accuracy, on_step=False, on_epoch=True)
        self.log('train_macro_f1', self.train_macro_f1, on_step=False, on_epoch=True)
        self.log('train_auc', self.train_auc, on_step=False, on_epoch=True)
        self.log('train_macro_acc_step', self.train_macro_accuracy, on_step=True, on_epoch=True)
        # Log loss
        self.log('train_loss', loss, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):

        img, tab, y, subject_id = batch
        y = y.to(torch.float32)

        y_pred = self(img, tab)

        loss_func = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.class_weight).float())
        
        loss = loss_func(y_pred, y.squeeze())
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
        
        self.val_results_df_all = pd.concat([self.val_results_df_all , self.val_results_df], ignore_index=True)
        
        print("val model.py")
        if BATCH_SIZE == 1:
            
            self.val_accuracy(torch.unsqueeze(y_pred_tag, 0), y)
            
            self.val_macro_accuracy(torch.unsqueeze(y_pred_tag, 0), y)
            self.val_macro_f1(torch.unsqueeze(y_pred_tag, 0), y)
            self.val_auc(torch.unsqueeze(y_pred_tag, 0), y)
        else:
            self.val_accuracy(y_pred_tag, y)
            
            self.val_macro_accuracy(y_pred_tag, y)
            self.val_macro_f1(y_pred_tag, y)
            self.val_auc(y_pred_tag, y)
        
        self.log('val_acc_step', self.val_accuracy, on_step=False, on_epoch=True)
        self.log('val_macro_acc_step', self.val_macro_accuracy, on_step=True, on_epoch=True)
        self.log('val_macro_f1', self.val_macro_f1, on_step=False, on_epoch=True)
        self.log('val_auc', self.val_auc, on_step=False, on_epoch=True)

        # Log loss
        self.log('val_loss', loss, on_step=True, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):

        img, tab, y, subject_id = batch
        y = y.to(torch.float32)

        y_pred = self(img, tab)

        loss_func = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.class_weight).float())

        loss = loss_func(y_pred, y.squeeze())        # loss = F.binary_cross_entropy(torch.sigmoid(y_pred), y.squeeze(), pos_weights = )
        
        y_pred_tag = torch.round(torch.sigmoid(y_pred))
                
        if BATCH_SIZE == 1:
            
            self.test_accuracy(torch.unsqueeze(y_pred_tag, 0), y)
            
            self.test_macro_accuracy(torch.unsqueeze(y_pred_tag, 0), y)
            self.test_macro_f1(torch.unsqueeze(y_pred_tag, 0), y)
            self.test_auc(torch.unsqueeze(y_pred_tag, 0), y)
        else:
            self.test_accuracy(y_pred_tag, y)
            
            self.test_macro_accuracy(y_pred_tag, y)
            self.test_macro_f1(y_pred_tag, y)
            self.test_auc(y_pred_tag, y)
        
        self.log('test_acc_step', self.test_accuracy, on_step=True, on_epoch=False)
        self.log('test_macro_acc_step', self.test_macro_accuracy, on_step=True, on_epoch=True)
        self.log('test_macro_f1', self.test_macro_f1, on_step=False, on_epoch=True)
        self.log('test_auc', self.test_auc, on_step=False, on_epoch=True)
        self.log("test loss", loss)

        return loss
        
    
    def training_epoch_end(self, outs):
        
        filename_out = '/home/users/ewesel/results/train_out_' + str(self.current_epoch) + '_' + TARGET + '_' + self.trainer.logger.experiment.name + '.csv'
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cpu":
            filename_out = '/Users/emilywesel/Desktop/NCANDA/train_out_' + str(self.current_epoch) + '_' + TARGET + '_' + self.trainer.logger.experiment.name + '.csv'

        self.train_results_df_all.to_csv(filename_out)

        # Clear the dataframe so the new epoch can start fresh
        self.train_results_df_all = pd.DataFrame(columns=self.results_column_names)
        # log epoch metric
        self.log('train_acc_epoch', self.train_accuracy)
        self.log('train_macro_acc_epoch', self.train_macro_accuracy)
        self.log('train_f1', self.train_macro_f1)
        self.log('train_auc', self.train_auc)
        
    
    def validation_epoch_end(self, outputs):
        # log epoch metric

        filename_out = '/home/users/ewesel/results/val_out_' + str(self.current_epoch) + '_' + TARGET + '_' + self.trainer.logger.experiment.name + '.csv'
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cpu":
            filename_out = '/Users/emilywesel/Desktop/NCANDA/val_out_' + str(self.current_epoch) + '_' + TARGET + '_' + self.trainer.logger.experiment.name + '.csv'

        self.val_results_df_all.to_csv(filename_out)

        # Clear the dataframe so the new epoch can start fresh
        self.val_results_df_all = pd.DataFrame(columns=self.results_column_names)
        self.log('val_macro_f1', self.val_macro_f1)
        self.log('val_auc', self.val_auc)
        self.log('val_acc_epoch', self.val_accuracy)
        self.log('val_macro_acc_epoch', self.val_macro_accuracy)
        
