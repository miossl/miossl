import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from model_utils import ClassificationModel
from typing import Type, Any
from datetime import datetime
from utils import save_model, plot_features
from perf_metrics import get_performance_metrics

class Trainer(nn.Module):
    def __init__(self,
                 model: nn.Module,
                 datamodule: nn.Module,
                 train_epochs: int = 100,
                 lineval_epochs: int = 100,
                 lineval_optim: str = 'sgd',
                 lineval_lr: float = 0.01,
                 lineval_momentum: float = 0.9,
                 lineval_wd: float = 0.0,
                 lineval_lr_schedule: str = 'steplr',
                 finetune_epochs: int = 100,
                 finetune_optim: str = 'sgd',
                 finetune_lr: float = 0.01,
                 finetune_momentum: float = 0.9,
                 finetune_wd: float = 0.0,
                 finetune_lr_schedule: str = 'steplr',
                 max_epochs: int = 1000,
                 modelsavepath: str = os.getcwd(),
                 modelsaveinterval: int = 1,
                 resume: bool = False,
                 model_path: str = None,
                 show_valid_metrics: bool = False,
                 **kwargs: Any) -> None:
        super().__init__()
        self.model = model
        self.datamodule = datamodule
        self.train_epochs = train_epochs
        self.lineval_epochs = lineval_epochs
        self.lineval_optim = lineval_optim
        self.lineval_lr = lineval_lr
        self.lineval_momentum = lineval_momentum
        self.lineval_wd = lineval_wd
        self.lineval_lr_schedule = lineval_lr_schedule
        self.finetune_epochs = finetune_epochs
        self.finetune_optim = finetune_optim
        self.finetune_lr = finetune_lr
        self.finetune_momentum = finetune_momentum
        self.finetune_wd = finetune_wd
        self.finetune_lr_schedule = finetune_lr_schedule
        self.max_epochs = max_epochs
        self.modelsavepath = modelsavepath
        self.modelsaveinterval = modelsaveinterval
        self.resume = resume
        self.model_path = model_path
        self.model_name = self.model.model_name
        self.show_valid_metrics = show_valid_metrics
        
        self.ds_num_classes = None

        self.ft_fc_lr = kwargs['ft_fc_lr'] if kwargs.__contains__('ft_fc_lr') else self.finetune_lr
        
        self.datestr = datetime.today().strftime("%d-%m-%y-%H-%M-%S")

        self.datamodule.prepare_data()

        self.writer = SummaryWriter('/'.join(['runs','_'.join([self.model_name,
                                                               self.model.base_encoder_name,
                                                               self.datestr])]))

    def fit(self):
        if self.resume:
            state_dict = torch.load(self.model_path)
            self.model.load_state_dict(state_dict['model_state_dict'])
            self.model.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
            self.model.scheduler.load_state_dict(state_dict['scheduler_state_dict'])
            start_epoch = self.model.scheduler.last_epoch
        else:
            start_epoch = 0
            
        self.datamodule.setup(stage = 'train',pretrain = True)
        self.datamodule.setup(stage = 'valid',pretrain = True)
        self.train_loader = self.datamodule.train_dataloader(True)
        self.valid_loader = self.datamodule.valid_dataloader(True)
        if hasattr(self.model, 'max_training_steps'):
            self.model.max_training_steps = self.max_epochs*(len(self.train_loader))
        #self.optimizer, self.scheduler = self.model.configure_optimizers()
        self.train_losses, self.valid_losses = np.array([]), np.array([])

        for epoch in range(start_epoch, self.train_epochs):
            print("\nEpoch {}".format(epoch+1), flush = True)
            train_epoch_loss = self.train_epoch(self.model, self.train_loader, self.model.optimizer)
            self.train_losses = np.append(self.train_losses, train_epoch_loss)
            self.writer.add_scalar('Pretrain/Loss/train',train_epoch_loss, epoch)
            self.model.scheduler.step()
            features, labels, val_epoch_loss = self.valid_epoch(self.model, self.valid_loader)
            self.writer.add_scalar('Pretrain/Loss/valid',val_epoch_loss, epoch)
            self.valid_losses = np.append(self.valid_losses, val_epoch_loss)
            #fig = plot_metrics(self.train_losses, self.valid_losses, 'Loss')
            #self.writer.add_figure('Pretrain/metrics',fig,epoch)
            if (epoch+1) % self.modelsaveinterval == 0:
                self.final_model_save_path = save_model(self.model,
                                                       epoch+1,
                                                       self.modelsavepath,
                                                       '_'.join([self.model.model_name,
                                                                '{}.pt']))
                print(f"Model at epoch {epoch+1} saved at {self.final_model_save_path}")
                fig = plot_features(features, labels, self.datamodule.num_classes, epoch)
                self.writer.add_figure('Pretrain/TSNE-Features',fig,epoch)

        #saving the final model
        self.final_model_save_path = '/'.join([self.modelsavepath,
                                                '_'.join([self.model.model_name,
                                                          self.datestr,'final.pt'])])
        torch.save(self.model.state_dict(), self.final_model_save_path)
        print(f"Final Model at epoch {epoch+1} saved at {self.final_model_save_path}")

        #SAVING ONLY THE ENCODER
        self.final_net_save_path = '/'.join([self.modelsavepath,
                                            '_'.join([self.model.model_name,
                                                      self.datestr,
                                                      'final_net.pt'])])
        torch.save(self.model.net.state_dict(), self.final_net_save_path)
        print(f"Encoder of Final Model at epoch {epoch+1} saved at {self.final_net_save_path}")

    def linear_eval(self,
                    dsmodel,
                    patience: int = 5,
                    net_model_path: str = None
                   ) -> Any:

        '''
            other_metrics: Dictionary {'metric1_name' : metric1_function,
                                        'metric2_name' : metric2_function}
        '''
        if net_model_path is not None:
            dsmodel.load_state_dict(torch.load(net_model_path), strict = False)
        else:
            dsmodel.load_state_dict(torch.load(self.final_net_save_path), strict = False)

        metrics = self.downstream('linear_eval',
                                  dsmodel,
                                  True, 1.0,
                                  self.lineval_epochs,
                                  patience,
                                  self.lineval_optim,
                                  self.lineval_lr_schedule
                                 )
        return metrics

    def fine_tune(self,
                  dsmodel,
                  patience: int = 5,
                  net_model_path: str = None
                 ) -> Any:
        metrics_dict = {}

        for fracs in [0.01, 0.1, 0.25, 0.5, 1.0]:
            #LOAD FINAL MODEL
            if net_model_path is not None:
                dsmodel.load_state_dict(torch.load(net_model_path), strict = False)
            else:
                dsmodel.load_state_dict(torch.load(self.final_net_save_path), strict = False)

            metrics = self.downstream('fine_tune',
                                      dsmodel,
                                      False, fracs,
                                      self.finetune_epochs,
                                      patience,
                                      self.finetune_optim,
                                      self.finetune_lr_schedule
                                     )
            metrics_dict = {**metrics_dict, **metrics}
        return metrics_dict

    def train_epoch(self,
                    model: nn.Module,
                    dataloader: nn.Module,
                    optimizer: nn.Module) -> int:
        model.train()
        train_losses = 0
        with tqdm(dataloader, unit = 'batch', total = len(dataloader)) as tepoch:
            for step, batch in enumerate(tepoch):
                optimizer.zero_grad()
                train_loss = model.step('train',batch, step)
                #self.writer.add_scalar('pretrain/train_loss_step',train_loss)
                train_loss.backward()
                optimizer.step()
                tepoch.set_postfix(loss = train_loss.item())
                train_losses += train_loss.item()
        train_losses/=(step+1)
        return train_losses

    def valid_epoch(self,
                    model: nn.Module,
                    dataloader: nn.Module) -> int:
        model.eval()
        valid_losses = 0
        features = np.array([]).reshape((0,self.model.projector_out_dim))
        labels = np.array([])
        with torch.no_grad():
            with tqdm(dataloader, unit = 'batch', total = len(dataloader)) as vepoch:
                for step, batch in enumerate(vepoch):
                    feats, label, valid_loss = model.step('valid',batch, step)
                    features = np.append(features, feats, axis = 0)
                    labels = np.append(labels, label)
                    #self.writer.add_scalar('pretrain/train_loss_step',train_loss)
                    vepoch.set_postfix(loss = valid_loss.item())
                    valid_losses += valid_loss.item()
            valid_losses/=(step+1)
        return features, labels, valid_losses

    def downstream(self,
                   stage: str,
                   model: nn.Module,
                   linear_eval: str,
                   fracs: float = 1.0,
                   ds_epochs: int = 100,
                   patience: int = 5,
                   optim: str = 'sgd',
                   scheduler: str = 'steplr'
                  ) -> None:

        stage = stage
        model = model
        linear_eval = linear_eval
        #for p in model.parameters():
        #    p.requires_grad = False
        #if not linear_eval:
        #    for p in model.net.base_encoder.parameters():
        #        p.requires_grad = True

        fracs = fracs
        ds_epochs = ds_epochs
        patience = patience
        counter = 0

        optim = optim
        scheduler = scheduler

        lr = self.lineval_lr if linear_eval else self.finetune_lr
        momentum = self.lineval_momentum if linear_eval else self.finetune_momentum

        if linear_eval:
            for n,p in model.named_parameters():
                if 'fc' not in n:
                    p.requires_grad = False
                else:
                    p.requires_grad = True
        else:
            for p in model.parameters():
                p.requires_grad = True

        self.evaluation_model = model #
        self.datamodule.setup(stage = 'train',pretrain = False, fracs = fracs)
        self.datamodule.setup(stage = 'valid',pretrain = False)
        self.train_loader = self.datamodule.train_dataloader(False)
        self.valid_loader = self.datamodule.valid_dataloader(False)
        self.datamodule.setup(stage = 'test',pretrain = False)
        self.test_loader = self.datamodule.test_dataloader(False)

        # setting different learning rates for differetn parts of the model
        if not linear_eval:
            params_encoder = []
            params_fc = []
            encoder_lr = self.finetune_lr
            fc_lr = self.ft_fc_lr
            for n,p in self.evaluation_model.named_parameters():
                if 'fc' not in n:
                    params_encoder.append(p)
                else:
                    params_fc.append(p)
            enc_params = {'params':params_encoder,'lr':encoder_lr}
            fc_params = {'params':params_fc, 'lr':fc_lr}
            parameters = [enc_params, fc_params]
        else:
            parameters = [p for p in self.evaluation_model.parameters() if p.requires_grad]

        if optim == 'sgd':
            optimizer = torch.optim.SGD(parameters, lr = lr, momentum = momentum)

        #CHANGE SCHEDULER LATER
        if scheduler == 'steplr':
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                           step_size=1,
                                                           gamma=0.98,
                                                           last_epoch=-1,
                                                           verbose = True)
        elif scheduler == 'cosine':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                       T_max = ds_epochs,
                                                                       last_epoch=-1,
                                                                       verbose = True)
        self.train_losses, self.valid_losses = np.array([]), np.array([])
        self.train_accuracy, self.valid_accuracy = np.array([]), np.array([])
        
        if self.evaluation_model.classification_type != 'multi-class':
            if self.datamodule.num_classes == 2 and self.evaluation_model.classification_type == 'binary':
                self.ds_num_classes = 1
            elif self.datamodule.num_classes == 2 and self.evaluation_model.classification_type == 'multi-label':
                self.ds_num_classes = 2
        else:
            self.ds_num_classes = self.datamodule.num_classes

        for epoch in range(ds_epochs):
            print("\nEpoch {}".format(epoch+1), flush = True)
            train_epoch_loss, train_epoch_accuracy = self.train_ds_epoch(self.evaluation_model,
                                                                        self.train_loader,
                                                                        optimizer)

            self.writer.add_scalar('/'.join([stage,str(fracs).replace('.','p'),'Loss','train']),train_epoch_loss,epoch)
            self.writer.add_scalar('/'.join([stage,str(fracs).replace('.','p'),'Accuracy','train']),train_epoch_accuracy,epoch)
            self.train_losses = np.append(self.train_losses, train_epoch_loss)
            self.train_accuracy = np.append(self.train_accuracy, train_epoch_accuracy)
            lr_scheduler.step()
            val_epoch_loss, val_epoch_accuracy, preds, gts = self.valid_ds_epoch(self.evaluation_model,
                                                                    self.valid_loader)
            self.valid_losses = np.append(self.valid_losses, val_epoch_loss)
            self.valid_accuracy = np.append(self.valid_accuracy, val_epoch_accuracy)

            self.writer.add_scalar('/'.join([stage,str(fracs).replace('.','p'),'Loss','valid']),val_epoch_loss,epoch)
            self.writer.add_scalar('/'.join([stage,str(fracs).replace('.','p'),'Accuracy','valid']),val_epoch_accuracy,epoch)
            print("\nTrain Accuracy : {acc:.5f}, Train Loss : {loss:.5f}".format(acc = train_epoch_accuracy, loss = train_epoch_loss), flush = True)
            print("\nValid Accuracy : {acc:.5f}, Valid Loss : {loss:.5f}".format(acc = val_epoch_accuracy, loss = val_epoch_loss), flush = True)
            
            if self.show_valid_metrics:
                val_perf_metrics = get_performance_metrics(gts, preds, [str(c) for c in list(range(self.ds_num_classes))])
                print(val_perf_metrics)

            if val_epoch_loss <= min(self.valid_losses):
                counter = 0
                if linear_eval:
                    self.dsfilepath = '/'.join([self.modelsavepath,'_'.join([self.model_name, 'linear_eval', self.datestr, '.pt'])])
                    torch.save(self.evaluation_model.state_dict(), self.dsfilepath)
                else:
                    self.dsfilepath = '/'.join([self.modelsavepath,'_'.join([self.model_name, 'fine_tune', self.datestr, '.pt'])])
                    torch.save(self.evaluation_model.state_dict(), self.dsfilepath)
            else:
                counter+=1
                if counter>patience:
                    print("Stopping Early. No more patience left.")
                    break

        #fig = plot_metrics(self.train_losses, self.valid_losses, 'Loss')
        #fig = plot_metrics(self.train_accuracy, self.valid_accuracy, 'Accuracy')
        ## LOADING THE BEST MODEL
        self.evaluation_model.load_state_dict(torch.load(self.dsfilepath))
        test_loss, test_acc, preds, gts = self.valid_ds_epoch(self.evaluation_model, self.test_loader)
        
        print(':::::Saving Predictions and One Hot Ground Truth data to \'.npy\' files:::::::')
        np.save('test_set_preds.npy', preds)
        np.save('test_set_onehot_gt.npy', gts)
        
        print(':::::::::::::::::::::::Class-wise Performance Metrics:::::::::::::::::::::::::')
        if self.evaluation_model.classification_type != 'multi-class':
            test_perf_metrics = get_performance_metrics(gts, preds, [str(c) for c in list(range(self.ds_num_classes))])
            print(test_perf_metrics)
        else:
            print("\nTest Accuracy : {acc:.5f}, Test Loss : {loss:.5f}".format(acc = test_acc, loss = test_loss), flush = True)
        
        min_ind = np.argmin(self.valid_losses)
        val_loss_min_ind = self.valid_losses[min_ind]
        val_acc_min_ind = self.valid_accuracy[min_ind]
        return {'_'.join([stage,str(fracs).replace('.','p'),'val_loss']):val_loss_min_ind,
                '_'.join([stage,str(fracs).replace('.','p'),'val_acc']):val_acc_min_ind,
                '_'.join([stage,str(fracs).replace('.','p'),'test_loss']):test_loss,
                '_'.join([stage,str(fracs).replace('.','p'),'test_acc']):test_acc}

    def train_ds_epoch(self, model, dataloader, optimizer):
        model.train()
        train_losses = 0
        train_accuracy = 0
        with tqdm(dataloader, unit = 'batch', total = len(dataloader)) as tepoch:
            for step, batch in enumerate(tepoch):
                optimizer.zero_grad()
                train_loss, train_acc, _, _ = model.step(batch, step)
                #self.writer.add_scalar('pretrain/train_loss_step',train_loss)
                train_loss.backward()
                optimizer.step()
                tepoch.set_postfix(loss = train_loss.item(), acc = train_acc)
                train_losses += train_loss.item()
                train_accuracy += train_acc
        train_losses/=(step+1)
        train_accuracy/=(step+1)
        return train_losses, train_accuracy

    def valid_ds_epoch(self, model, dataloader):
        model.eval()
        valid_losses = 0
        valid_accuracy = 0
        preds = np.array([]).reshape((0,self.ds_num_classes))
        gts = np.array([]).reshape((0,self.ds_num_classes))
        with torch.no_grad():
            with tqdm(dataloader, unit = 'batch', total = len(dataloader)) as vepoch:
                for step, batch in enumerate(vepoch):
                    valid_loss, valid_acc, pred, gt = model.step(batch, step)
                    preds = np.append(preds, pred.numpy(), axis = 0)
                    if self.evaluation_model.classification_type == 'multi-class':
                        gt = torch.nn.functional.one_hot(gt, num_classes = self.ds_num_classes)
                    gts = np.append(gts, gt.numpy(), axis = 0)
                    #self.writer.add_scalar('pretrain/train_loss_step',train_loss)
                    vepoch.set_postfix(loss = valid_loss.item(), acc = valid_acc)
                    valid_losses += valid_loss.item()
                    valid_accuracy += valid_acc
            valid_losses/=(step+1)
            valid_accuracy/=(step+1)
        return valid_losses, valid_accuracy, preds, gts
