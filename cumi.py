# IMPORTS
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from torchvision import models

from model_utils import Identity, ProjectionHead
from lars import LARS
from schedulers import LinearWarmupCosineAnnealingLR
from losses import CUMILoss
from utils import seed_everything

seed_everything(16)

# MODEL

class CUMINet(nn.Module):
    def __init__(self,
                 base_encoder_name: str = 'resnet50',
                 projector_type: str = 'nonlinear',
                 proj_num_layers: int = 2,
                 projector_hid_dim: int = None,
                 projector_out_dim: int = 128,
                 proj_use_bn: bool = True,
                 proj_last_bn: bool = True,
                 data_dims: str = '224x224') -> None:
        super().__init__()
        self.base_encoder_name = base_encoder_name
        self.projector_type = projector_type
        self.proj_num_layers = proj_num_layers
        self.projector_out_dim = projector_out_dim
        self.data_dims = data_dims
        self.base_encoder = models.resnet50()
        dim_infeat = self.base_encoder.fc.in_features
        self.projector_hid_dim = projector_hid_dim
        if self.projector_hid_dim is None:
            self.projector_hid_dim = dim_infeat

        self.base_encoder.fc = Identity()
        if self.data_dims.split('x')[0] == '32':
            self.base_encoder.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), bias=False)
            self.base_encoder.maxpool = Identity()

        for p in self.base_encoder.parameters():
            p.requires_grad = True

        self.projector = ProjectionHead(in_features = dim_infeat,
                                        hidden_features = self.projector_hid_dim,
                                        out_features = self.projector_out_dim,
                                        head_type = self.projector_type,
                                        num_layers = self.proj_num_layers,
                                        use_bn = True,
                                        last_bn = True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.base_encoder(x)
        x = self.projector(x)
        return x

class CUMIModel(nn.Module):

    def __init__(self,
                 base_encoder_name: str = 'resnet50',
                 data_dims: str = '224x224',
                 optim: str = 'lars',
                 lr: float = 0.3,
                 momentum: float = 0.9,
                 temperature: float = 0.5,
                 n_temperature: float = None,
                 lambda_loss: float = 1.0,
                 weight_decay: float = 1e-6,
                 warmup_epochs: int = 10,
                 max_epochs: int = 1000,
                 warmup_start_lr: int = 0.0001,
                 eta_min: int = 0.0001,
                 pretrain_batch_size: int = 64,
                 other_batch_size: int = 32,
                 projector_type: str = 'nonlinear',
                 proj_num_layers: int = 2,
                 projector_hid_dim: int = None,
                 projector_out_dim: int = 128,
                 proj_use_bn: bool = True,
                 proj_last_bn: bool = True,
                 **kwargs) -> None:
        super().__init__()

        self.base_encoder_name = base_encoder_name
        self.data_dims = data_dims
        self.optim = optim
        self.lr = lr
        self.momentum = momentum
        self.temperature = temperature
        if n_temperature is not None:
            self.p_temperature = self.temperature
            self.n_temperature = n_temperature
        else:
            self.p_temperature = self.temperature
            self.n_temperature = self.temperature

        self.lambda_loss = lambda_loss
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        self.pretrain_batch_size = pretrain_batch_size
        self.other_batch_size = other_batch_size
        self.projector_hid_dim = projector_hid_dim
        self.projector_out_dim = projector_out_dim
        self.proj_num_layers = proj_num_layers
        self.projector_type = projector_type
        self.proj_use_bn = proj_use_bn
        self.proj_last_bn = proj_last_bn

        self.net = CUMINet(self.base_encoder_name,
                             self.projector_type,
                             self.proj_num_layers,
                             self.projector_hid_dim,
                             self.projector_out_dim,
                             self.proj_use_bn,
                             self.proj_last_bn,
                             self.data_dims).to('cuda:0')

        self.criterion = CUMILoss(batch_size = self.pretrain_batch_size,
                                  p_temperature = self.p_temperature,
                                  n_temperature = self.n_temperature,
                                  lambda_loss = self.lambda_loss)

        self.optimizer, self.scheduler = self.configure_optimizers()

    @property
    def model_name(self) -> str:
        return 'cumi'

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

    def configure_optimizers(self):

        if self.optim == 'lars':
            params = []
            param_names = []
            for n,p in self.net.named_parameters():
                params.append(p)
                param_names.append(n)
            parameters = [{'params':params,'param_names':param_names}]
            optimizer = LARS(parameters,
                             lr = self.lr,
                             weight_decay = self.weight_decay,
                             exclude_from_weight_decay=["batch_normalization", "bias"])

        scheduler = LinearWarmupCosineAnnealingLR(optimizer,
                                                  self.warmup_epochs,
                                                  self.max_epochs,
                                                  self.warmup_start_lr,
                                                  self.eta_min)
        return optimizer, scheduler

    def step(self, stage, batch, batch_idx):
        x1, x2, y = [b.to('cuda:0') for b in batch]
        #  pass throught net
        x1 = self.net(x1)
        x2 = self.net(x2)
        loss = self.criterion(x1,x2)
        #  self.log('train_loss_ssl',loss, on_epoch = True, logger = True)
        if stage == 'train':
            return loss
        else:
            return x1.cpu().numpy(), y.cpu().numpy(), loss #.cpu().item()
