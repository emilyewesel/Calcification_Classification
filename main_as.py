import nibabel as nib
import numpy as np
import pandas as pd


import wandb
import os
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
# from medcam import medcam
from scipy.interpolate import interpn
import numpy as np
from conv3D.model import AdniModel
# from dataset import odule
from unimodal_dataset import ASDataModule


from ResNet.model import ResNetModel
from SiameseNet.model import SiameseResNetModel

import torch.multiprocessing as mp

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = device.type


def main_conv3d(wandb, wandb_logger):
    '''
    main function to run the conv3d architecture
    '''
    seed_everything(23)
    # get the model
    model = AdniModel()

    # load the data
    data = ASDataModule()

    # Optional
    wandb.watch(model, log="all")

    # train the network
    trainer = Trainer(max_epochs=15, logger=wandb_logger, log_every_n_steps=1, accelerator=device, devices=1)
    trainer.fit(model, data)


def main_resnet(wandb, wandb_logger):
    '''
    main function to run the resnet architecture
    '''
    seed_everything(23)
   
    # load the data
    data = ASDataModule()
     # ge the model
    data.prepare_data()

    # ge the model
    model = ResNetModel(class_weights=data.class_weight)

    # Optional
    wandb.watch(model, log="all")

    # train the network
    # if device == "cpu":
    #     data = data.to("cpu")
    trainer = Trainer(max_epochs=50, logger=wandb_logger, log_every_n_steps=1, accelerator=device, devices=1)
    trainer.fit(model, data)

def main_siamese(wandb, wandb_logger):
    '''
    main function to run the resnet architecture
    '''
    seed_everything(23)
   
    # load the data
    data = ASDataModule()
     # ge the model
    data.prepare_data()

    # ge the model
    model = SiameseResNetModel(class_weights=data.class_weight)

    # Optional
    wandb.watch(model, log="all")

    # train the network
    # if device == "cpu":
    #     data = data.cpu()
    trainer = Trainer(max_epochs=50, logger=wandb_logger, log_every_n_steps=1, accelerator=device, devices=1)
    trainer.fit(model, data)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)


    # create wandb objects to track runs
    # wandb.init(project="ncanda-imaging")
    # wandb.config = {
    #     "learning_rate": 1e-4,
    #     "epochs": 5,
    #     "batch_size": 1
    # }

    wandb_logger = WandbLogger(wandb.init(project="ncanda-emily", entity="ewesel"))

    # run conv3d
    # main_conv3d(wandb, wandb_logger)

    # run siamense
    # main_siamese(wandb, wandb_logger)
    
    #run resnet
    main_resnet(wandb, wandb_logger)


