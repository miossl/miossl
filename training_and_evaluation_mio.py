# -*- coding: utf-8 -*-

"""## Training"""

import os
from argparse import ArgumentParser
from tabulate import tabulate
from datetime import datetime
import torch
import torch.nn as nn
import torchvision
import miossl.mio as mio
from miossl.dataloader_modules import CIFAR10ArrayDataModule, STL10DataModule, CIFAR100DataModule
from miossl.trainer import Trainer
from miossl.utils import run_command
from miossl.model_utils import ClassificationModel
from miossl.model_transforms import MIOTransform

"""## Declaring the model"""

model = mio.MIOModel(optim = 'lars', 
                       proj_last_bn = True, 
                       warmup_epochs = 10, 
                       pretrain_batch_size = 128, 
                       lr = 1.5, 
                       data_dims = '32x32', 
                       max_epochs = 1000, 
                       temperature = 0.5,
                       lambda_loss = 0.0,
                       proj_num_layers = 2,
                       projector_hid_dim = 2048,
                       projector_out_dim = 128)

"""## Transformations for Augmentation"""

# 's' is the scaling factor for brightness, contrast, hue and saturation. 'l' is the dimension of the input images as well as the output image dimension.
transforms = MIOTransform(s = 0.5, 
                          l = 32)

"""## Datamodules


"""

# If the dataset is already downloaded, set download = False and set dataset_path to the location of the dataset folder 'cifar-10-python'

dm = CIFAR10ArrayDataModule(pretrain_batch_size = 128, 
                            other_batch_size = 32, 
                            download = True, 
                            dataset_path = '/content/cifar-10-python', 
                            transformations = transforms)

# # If the dataset is already downloaded, set download = False and set dataset_path to the location of the dataset folder 'cifar-10-python'
# dm = STL10DataModule(pretrain_batch_size = 128, 
#                      other_batch_size = 32, 
#                      download = True,
#                      dataset_path = '/content/stl10_binary', 
#                      transformations = transforms)

# # If the dataset is already downloaded, set download = False and set dataset_path to the location of the dataset folder 'cifar-10-python'
# # if fine_labels = True, then total number of classes will be 100, if fine_labels is set to False, total number of classes is 20
# dm = CIFAR100DataModule(pretrain_batch_size = 128, 
#                         other_batch_size = 32, 
#                         fine_labels = True, 
#                         download = True, 
#                         dataset_path = '/content/cifar-100-python/', 
#                         transformations = transforms)



"""## Initialize the Trainer

- Trainer will download the dataset is download is set to True in the previous step. 
"""

trainer = Trainer(model = model, 
                  datamodule = dm, 
                  train_epochs = 250, 
                  modelsaveinterval = 25, 
                  max_epochs = 1000) #, resume = True, model_path = '') If training needs to be resumed

"""## Start Training"""

trainer.fit()

"""## Declare the model for linear classification"""

ds_model = ClassificationModel('resnet50',dm.num_classes, '32x32').to('cuda:0')

"""## Linear Evaluation training"""

# LINEAR EVALUATION
lin_eval_metrics = trainer.linear_eval(ds_model, patience=50) #, net_model_path = '/content/moco_29-10-21-04-29-53_final_net.pt')

"""## Print the Linear Evaluation Metrics"""

print(lin_eval_metrics)
