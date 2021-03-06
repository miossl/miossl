import torch
import torch.nn as nn
from typing import Any
from torch import Tensor
from torchvision import models

class Identity(nn.Module):
    def __init__(self) -> None:
        super(Identity, self).__init__()

    def forward(self, x: Tensor)-> Tensor:
        return x

class LinearLayer(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 use_bias: bool = True,
                 use_bn: bool = False,
                 **kwargs: Any) -> None:
        super(LinearLayer, self).__init__(**kwargs)

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.use_bn = use_bn

        self.linear_layer = nn.Linear(self.in_features,
                                self.out_features,
                                bias = self.use_bias and not self.use_bn)
        if self.use_bn:
            self.bn_layer = nn.BatchNorm1d(self.out_features)

    def forward(self,x: Tensor) -> Tensor:
        x = self.linear_layer(x)
        if self.use_bn:
            x = self.bn_layer(x)
        return x

class ProjectionHead(nn.Module):
    def __init__(self,
                 in_features: int = None,
                 hidden_features: int = None,
                 out_features: int = None,
                 head_type: str = 'nonlinear',
                 num_layers: int = 2,
                 use_bn: bool = True,
                 **kwargs: Any) -> None:
        super(ProjectionHead,self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.head_type = head_type
        self.num_layers = num_layers
        self.use_bn = use_bn
        # FOR BYOL
        self.last_bn = kwargs['last_bn'] if kwargs.__contains__('last_bn') else self.use_bn

        if self.head_type == 'linear':
            self.mlp = LinearLayer(self.in_features,self.out_features, False, self.use_bn)
        elif self.head_type == 'nonlinear':
            self.layers = []
            self.layers.append(LinearLayer(self.in_features,self.hidden_features, use_bias = True, use_bn = self.use_bn))
            self.layers.append(nn.ReLU())
            for i in range(self.num_layers-2):
                self.layers.append(LinearLayer(self.hidden_features,self.hidden_features, use_bias = True, use_bn = self.use_bn))
                self.layers.append(nn.ReLU())
            self.layers.append(LinearLayer(self.hidden_features,self.out_features, use_bias = True, use_bn = self.last_bn))

            self.mlp = nn.Sequential(*self.layers)

    def forward(self,x: Tensor) -> Tensor:
        x = self.mlp(x)
        return x

class ClassificationModel(nn.Module):
    def __init__(self,
                 model_name: str,
                 num_classes: int,
                 data_dims: str,
                 classification_type: str = 'multi-class',
                 **kwargs: Any) -> None:
        super(ClassificationModel, self).__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.data_dims = data_dims
        self.classification_type = classification_type

        self.base_encoder = models.resnet50(num_classes = self.num_classes)
        if self.data_dims.split('x')[0] == '32':
            self.base_encoder.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), bias=False)
            self.base_encoder.maxpool = Identity()
        
        self.bin_pos_wts = kwargs['bin_pos_wts'] if kwargs.__contains__('bin_pos_wts') else 1.0
        if self.classification_type == 'multi-class':
            self.criterion = nn.CrossEntropyLoss()
        elif self.classification_type in ['binary','multi-label']:
            self.criterion = nn.BCEWithLogitsLoss(pos_weight = self.bin_pos_wts)

    def forward(self, x:Tensor) -> Tensor:
        return self.base_encoder(x)

    def step(self, batch, batch_idx):
        x, y = [b.to('cuda:0') for b in batch]
        # pass throught net
        z = self.base_encoder(x)
        
        if self.classification_type != 'multi-class':
            y = y.to(dtype = torch.float)
        if self.classification_type == 'binary':
            y = y.view((-1,1))
            
        loss = self.criterion(z,y)
        
        if self.classification_type == 'multi-class':
            preds = torch.exp(z.cpu().data)/torch.sum(torch.exp(z.cpu().data), dim = 1, keepdims = True)
            _, preds = torch.max(preds, dim = 1)
        else:
            preds = (z.cpu().data >= 0.5)
        
        acc = (preds == y.cpu().data).to(dtype = torch.float64).mean().item()
        #self.log('train_loss_ssl',loss, on_epoch = True, logger = True)
        return loss, acc, z.cpu().data, y.cpu().data

