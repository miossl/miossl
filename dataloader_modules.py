import numpy as np
import pandas as pd
import os, requests, copy, random
import imageio
from typing import Optional, Any
from tqdm import tqdm
#from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from model_transforms import *
from extraction import extract_archive
from utils import unpickle

class SSLArrayDataset(Dataset):
    def __init__(self, phase, array, labels, mean, std, transformations = None):
        super().__init__()
        self.phase = phase
        self.imgarr = copy.deepcopy(array)
        self.labels = copy.deepcopy(labels)
        self.normalize = transforms.Normalize(mean,std)
        self.transforms = transformations

    def __len__(self):
        return self.imgarr.shape[0]

    def __getitem__(self,idx):
        x = self.imgarr[idx]
        x1, x2 = self.augment(torch.from_numpy(x))
        y = self.labels[idx].astype(np.int64)
        return x1, x2, y
    #shuffles the dataset at the end of each epoch
    def on_epoch_end(self):
        reidx = random.sample(population = list(range(self.__len__())),k = self.__len__())
        self.imgarr = self.imgarr[reidx]
        self.labels = self.labels[reidx]
    #applies randomly selected augmentations to each clip (same for each frame in the clip)
    def augment(self, x):
        if self.transforms is not None:
            x1, x2 = self.transforms(x)
        else:
            x1, x2 = x, x
        x1 = self.normalize(x1)
        x2 = self.normalize(x2)
        return x1, x2

class SLArrayDataset(Dataset):
    def __init__(self, phase, array, labels, mean, std, transformations = None, fracs = 1.0):
        super().__init__()
        self.phase = phase
        self.imgarr = copy.deepcopy(array)
        self.labels = copy.deepcopy(labels)
        self.fracs = fracs
        if self.fracs < 1.0:
            self.imgarr, _, self.labels, _ = train_test_split(self.imgarr, self.labels, test_size = 1 - fracs)
        self.normalize = transforms.Normalize(mean, std)
        self.transforms = transformations

    def __len__(self):
        return self.imgarr.shape[0]

    def __getitem__(self,idx):
        x = self.imgarr[idx]
        x = self.augment(torch.from_numpy(x))
        y = self.labels[idx].astype(np.int64)
        return x, y
    #shuffles the dataset at the end of each epoch
    def on_epoch_end(self):
        reidx = random.sample(population = list(range(self.__len__())),k = self.__len__())
        self.imgarr = self.imgarr[reidx]
        self.labels = self.labels[reidx]
    #applies randomly selected augmentations to each clip (same for each frame in the clip)
    def augment(self, x):
        if self.transforms is not None:
            x = self.transforms(x)
        x = self.normalize(x)
        return x

class SSLDataFrameDataset(Dataset):
    def __init__(self, phase, df, mean, std, transformations = None):
        super().__init__()
        self.phase = phase
        self.df = df
        self.normalize = transforms.Normalize(mean,std)
        self.transforms = transformations

    def __len__(self):
        return len(self.df)

    def __getitem__(self,idx):
        index = idx
        done = False
        while not done:
            try:
                x = imageio.imread(self.df['filename'].iloc[index])
                done = True
            except:
                index += 1
        if len(x.shape)==2:
            x = np.repeat(np.expand_dims(x, 2), 3, 2)
        x = np.transpose(x, (2,0,1))
        x = x/255.0
        x1, x2 = self.augment(torch.from_numpy(x))
        y = self.df['label'].iloc[idx] #.astype(np.int64)
        return x1.to(dtype = torch.float), x2.to(dtype = torch.float), y
    
    #shuffles the dataset at the end of each epoch
    def on_epoch_end(self):
        idx = np.random.permutation(self.df.index.values)
        self.df = self.df.reindex(idx)
        self.df = self.df.reset_index(drop=True)
    
    #applies randomly selected augmentations to each clip (same for each frame in the clip)
    def augment(self, x):
        if self.transforms is not None:
            x1, x2 = self.transforms(x)
        else:
            x1, x2 = x, x
        x1 = self.normalize(x1)
        x2 = self.normalize(x2)
        return x1, x2

class SLDataFrameDataset(Dataset):
    def __init__(self, phase, df, mean, std, transformations = None, fracs = 1.0):
        super().__init__()
        self.phase = phase
        self.df = df
        self.fracs = fracs
        if self.fracs < 1.0:
            self.imgarr, _, self.labels, _ = train_test_split(self.imgarr, self.labels, test_size = 1 - fracs)
        self.normalize = transforms.Normalize(mean, std)
        self.transforms = transformations

    def __len__(self):
        return len(self.df)

    def __getitem__(self,idx):
        index = idx
        done = False
        while not done:
            try:
                x = imageio.imread(self.df['filename'].iloc[index])
                done = True
            except:
                index += 1
        
        if len(x.shape)==2:
            x = np.repeat(np.expand_dims(x, 2), 3, 2)
        x = np.transpose(x, (2,0,1))     
        x = x/255.0   
        x = self.augment(torch.from_numpy(x))
        y = self.df['label'].iloc[idx] #.astype(np.int64)
        return x.to(dtype = torch.float), y

    #shuffles the dataset at the end of each epoch
    def on_epoch_end(self):
        idx = np.random.permutation(self.df.index.values)
        self.df = self.df.reindex(idx)
        self.df = self.df.reset_index(drop=True)

    #applies randomly selected augmentations to each clip (same for each frame in the clip)
    def augment(self, x):
        if self.transforms is not None:
            x = self.transforms(x)
        x = self.normalize(x)
        return x

class CIFAR10DFDataModule(nn.Module):
    def __init__(self,
                 pretrain_batch_size: int = 64,
                 other_batch_size: int = 32,
                 download: bool = False,
                 dataset_path: str = None,
                 transformations = None
                 ) -> None:
        super().__init__()
        self.pretrain_batch_size = pretrain_batch_size
        self.other_batch_size = other_batch_size
        self.download = download
        self.dataset_path = dataset_path
        self.transforms = transformations

    @property
    def num_classes(self) -> int:
        return 10
    
    def prepare_data(self):
        #DATA READING AND SPLITTING
        #-----training data
        self.lab_dict = {0:'airplane',1:'automobile',2:'bird',3:'cat',4:'deer',5:'dog',6:'frog',7:'horse',8:'ship',9:'truck'}
        # splitting training and validation data
        self.train_df = pd.read_csv('/'.join([self.dataset_path,'train.csv']))
        self.train_df, self.valid_df = train_test_split(self.train_df, test_size = 0.2)
        self.train_df = self.train_df.reset_index(drop = True)
        self.valid_df = self.valid_df.reset_index(drop = True)

        self.MEAN = np.array([0.485, 0.456, 0.406]).reshape((1,3,1,1))
        self.STD = np.array([0.229, 0.224, 0.225]).reshape((1,3,1,1))
        #----TEST IMAGES
        self.test_df = pd.read_csv('/'.join([self.dataset_path, 'test.csv']))

    def setup(self, stage: Optional[str] = None, pretrain : Optional[bool] = True, fracs: Optional[float] = 1.0):
        #transforms
        if stage == 'train':
            self.train_transforms = self.transforms
            if pretrain:
                self.traingen = SSLDataFrameDataset('train', self.train_df, self.MEAN, self.STD, self.train_transforms)
            else:
                self.traingen = SLDataFrameDataset('train', self.train_df, self.MEAN, self.STD, transforms.RandomResizedCrop(32,(0.8,1.0)), fracs)
        if stage == 'valid':
            self.valid_transforms = None #transforms.Compose([transforms.Normalize(self.MEAN, self.STD)])
            if pretrain:
                self.validgen = SSLDataFrameDataset('valid', self.valid_df, self.MEAN, self.STD, self.valid_transforms) #torchvision.transforms.RandomResizedCrop(32,(0.8,1.0))
            else:
                self.validgen = SLDataFrameDataset('valid', self.valid_df, self.MEAN, self.STD, self.valid_transforms) #torchvision.transforms.RandomResizedCrop(32,(0.8,1.0))

        if stage == 'test':
            self.test_transforms = None #transforms.Compose([transforms.Normalize(self.MEAN, self.STD)])
            if pretrain:
                self.testgen = SSLDataFrameDataset('test', self.test_df, self.MEAN, self.STD, self.test_transforms)
            else:
                self.testgen = SLDataFrameDataset('test', self.test_df, self.MEAN, self.STD, self.test_transforms)

    def train_dataloader(self, pretrain : Optional[bool] = True):
        if pretrain:
            trainloader = DataLoader(self.traingen, batch_size = self.pretrain_batch_size, shuffle = True, drop_last = True)
        else:
            trainloader = DataLoader(self.traingen, batch_size = self.other_batch_size, shuffle = True, drop_last = True)
        return trainloader
    def valid_dataloader(self, pretrain : Optional[bool] = True):
        if pretrain:
            validloader = DataLoader(self.validgen, batch_size = self.pretrain_batch_size, drop_last = True)
        else:
            validloader = DataLoader(self.validgen, batch_size = self.other_batch_size, drop_last = True)
        return validloader
    def test_dataloader(self, pretrain : Optional[bool] = True):
        if pretrain:
            testloader = DataLoader(self.testgen, batch_size = self.pretrain_batch_size, drop_last = True)
        else:
            testloader = DataLoader(self.testgen, batch_size = self.other_batch_size, drop_last = True)
        return testloader


class CIFAR10ArrayDataModule(nn.Module):
    def __init__(self,
                 pretrain_batch_size: int = 64,
                 other_batch_size: int = 32,
                 download: bool = False,
                 dataset_path: str = None,
                 transformations = None
                 ) -> None:
        super().__init__()
        self.pretrain_batch_size = pretrain_batch_size
        self.other_batch_size = other_batch_size
        self.download = download
        self.dataset_path = dataset_path
        self.transforms = transformations

    @property
    def num_classes(self) -> int:
        return 10

    def prepare_data(self):
        if self.download:
            url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
            r = requests.get(url, stream = True, allow_redirects = True)
            filename = url.split('/')[-1]
            with open(filename,'wb') as f:
                for chunk in tqdm(r.iter_content(chunk_size = 1024)):
                    if chunk:
                        _ = f.write(chunk)

            self.dataset_path = extract_archive('/'.join([os.getcwd(),filename]))

        #DATA READING AND SPLITTING
        #-----training data
        self.lab_dict = {0:'airplane',1:'automobile',2:'bird',3:'cat',4:'deer',5:'dog',6:'frog',7:'horse',8:'ship',9:'truck'}
        train_files = ['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5']
        images = np.array([],dtype=np.uint8).reshape((0,3072))
        labels = np.array([])
        for tf in train_files:
            data_dict = unpickle(self.dataset_path + '/cifar-10-batches-py/'+tf)
            data = data_dict[b'data']
            images = np.append(images,data,axis=0)
            labels = np.append(labels,data_dict[b'labels'])
        #print(images.shape, labels.shape)
        images = images.reshape((-1,3,32,32)).astype(np.float32)
        images = images/255.0
        # splitting training and validation data
        self.trimages, self.valimages, self.trlabels, self.vallabels = train_test_split(images, labels, test_size = 0.2)

        self.MEAN = np.mean(self.trimages, axis = (0,2,3), keepdims = False)
        self.STD = np.std(self.trimages, axis = (0,2,3), keepdims = False)
        #----TEST IMAGES
        self.testimages = np.array([],dtype=np.uint8).reshape((0,3072))
        self.testlabels = np.array([])
        data_dict = unpickle(self.dataset_path + '/cifar-10-batches-py/test_batch')
        data = data_dict[b'data']
        self.testimages = np.append(self.testimages,data,axis=0)
        self.testlabels = np.append(self.testlabels,data_dict[b'labels'])
        #print(testimages.shape, testlabels.shape)
        self.testimages = self.testimages.reshape((-1,3,32,32)).astype(np.float32)
        self.testimages = self.testimages/255.0

    def setup(self, stage: Optional[str] = None, pretrain : Optional[bool] = True, fracs: Optional[float] = 1.0):
        #transforms
        if stage == 'train':
            self.train_transforms = self.transforms
            if pretrain:
                self.traingen = SSLArrayDataset('train', self.trimages, self.trlabels, self.MEAN, self.STD, self.train_transforms)
            else:
                self.traingen = SLArrayDataset('train', self.trimages, self.trlabels, self.MEAN, self.STD, transforms.RandomResizedCrop(32,(0.8,1.0)), fracs)
        if stage == 'valid':
            self.valid_transforms = None #transforms.Compose([transforms.Normalize(self.MEAN, self.STD)])
            if pretrain:
                self.validgen = SSLArrayDataset('valid', self.valimages, self.vallabels, self.MEAN, self.STD, self.valid_transforms) #torchvision.transforms.RandomResizedCrop(32,(0.8,1.0))
            else:
                self.validgen = SLArrayDataset('valid', self.valimages, self.vallabels, self.MEAN, self.STD, self.valid_transforms) #torchvision.transforms.RandomResizedCrop(32,(0.8,1.0))

        if stage == 'test':
            self.test_transforms = None #transforms.Compose([transforms.Normalize(self.MEAN, self.STD)])
            if pretrain:
                self.testgen = SSLArrayDataset('test', self.testimages, self.testlabels, self.MEAN, self.STD, self.test_transforms)
            else:
                self.testgen = SLArrayDataset('test', self.testimages, self.testlabels, self.MEAN, self.STD, self.test_transforms)

    def train_dataloader(self, pretrain : Optional[bool] = True):
        if pretrain:
            trainloader = DataLoader(self.traingen, batch_size = self.pretrain_batch_size, shuffle = True, drop_last = True)
        else:
            trainloader = DataLoader(self.traingen, batch_size = self.other_batch_size, shuffle = True, drop_last = True)
        return trainloader
    def valid_dataloader(self, pretrain : Optional[bool] = True):
        if pretrain:
            validloader = DataLoader(self.validgen, batch_size = self.pretrain_batch_size, drop_last = True)
        else:
            validloader = DataLoader(self.validgen, batch_size = self.other_batch_size, drop_last = True)
        return validloader
    def test_dataloader(self, pretrain : Optional[bool] = True):
        if pretrain:
            testloader = DataLoader(self.testgen, batch_size = self.pretrain_batch_size, drop_last = True)
        else:
            testloader = DataLoader(self.testgen, batch_size = self.other_batch_size, drop_last = True)
        return testloader

class CIFAR100DataModule(nn.Module):
    def __init__(self,
                 pretrain_batch_size: int = 64,
                 other_batch_size: int = 32,
                 fine_labels: bool = True,
                 download: bool = False,
                 dataset_path: str = None,
                 transformations = None
                 ) -> None:
        super().__init__()
        self.pretrain_batch_size = pretrain_batch_size
        self.other_batch_size = other_batch_size
        self.fine_labels = fine_labels
        if self.fine_labels:
            self.label_type = b'fine_labels'
        else:
            self.label_type = b'coarse_labels'
        self.download = download
        self.dataset_path = dataset_path
        self.transforms = transformations

    @property
    def num_classes(self) -> int:
        return 100

    def prepare_data(self):
        if self.download:
            url = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
            r = requests.get(url, stream = True, allow_redirects = True)
            filename = url.split('/')[-1]
            with open(filename,'wb') as f:
                for chunk in tqdm(r.iter_content(chunk_size = 1024)):
                    if chunk:
                        _ = f.write(chunk)

            self.dataset_path = extract_archive('/'.join([os.getcwd(),filename]))

        #DATA READING AND SPLITTING
        #-----training data
        meta_data = unpickle(self.dataset_path + '/cifar-100-python/meta')
        self.fine_labels = [s.decode('latin1') for s in meta_data[b'fine_label_names']]
        self.coarse_labels = [s.decode('latin1') for s in meta_data[b'coarse_label_names']]

        self.fine_lab_dict = dict(zip(np.arange(100),self.fine_labels))
        self.coarse_lab_dict = dict(zip(np.arange(20), self.coarse_labels))

        #train_files = ['train']
        images = np.array([],dtype=np.uint8).reshape((0,3072))
        labels = np.array([])
        data_dict = unpickle(self.dataset_path + '/cifar-100-python/train')
        
        data = data_dict[b'data']
        images = np.append(images,data,axis=0)
        labels = np.append(labels,data_dict[self.label_type])
        #print(images.shape, labels.shape)
        images = images.reshape((-1,3,32,32)).astype(np.float32)
        images = images/255.0
        # splitting training and validation data
        self.trimages, self.valimages, self.trlabels, self.vallabels = train_test_split(images, labels, test_size = 0.2)

        self.MEAN = np.mean(self.trimages, axis = (0,2,3), keepdims = False)
        self.STD = np.std(self.trimages, axis = (0,2,3), keepdims = False)
        #----TEST IMAGES
        self.testimages = np.array([],dtype=np.uint8).reshape((0,3072))
        self.testlabels = np.array([])
        data_dict = unpickle(self.dataset_path + '/cifar-100-python/test')
        
        data = data_dict[b'data']
        self.testimages = np.append(self.testimages,data,axis=0)
        self.testlabels = np.append(self.testlabels,data_dict[self.label_type])
        #print(testimages.shape, testlabels.shape)
        self.testimages = self.testimages.reshape((-1,3,32,32)).astype(np.float32)
        self.testimages = self.testimages/255.0

    def setup(self, stage: Optional[str] = None, pretrain : Optional[bool] = True, fracs: Optional[float] = 1.0):
        #transforms
        if stage == 'train':
            self.train_transforms = self.transforms
            if pretrain:
                self.traingen = SSLArrayDataset('train', self.trimages, self.trlabels, self.MEAN, self.STD, self.train_transforms)
            else:
                self.traingen = SLArrayDataset('train', self.trimages, self.trlabels, self.MEAN, self.STD, transforms.RandomResizedCrop(32,(0.8,1.0)), fracs)
        if stage == 'valid':
            self.valid_transforms = None #transforms.Compose([transforms.Normalize(self.MEAN, self.STD)])
            if pretrain:
                self.validgen = SSLArrayDataset('valid', self.valimages, self.vallabels, self.MEAN, self.STD, self.valid_transforms) #torchvision.transforms.RandomResizedCrop(32,(0.8,1.0))
            else:
                self.validgen = SLArrayDataset('valid', self.valimages, self.vallabels, self.MEAN, self.STD, self.valid_transforms) #torchvision.transforms.RandomResizedCrop(32,(0.8,1.0))

        if stage == 'test':
            self.test_transforms = None #transforms.Compose([transforms.Normalize(self.MEAN, self.STD)])
            if pretrain:
                self.testgen = SSLArrayDataset('test', self.testimages, self.testlabels, self.MEAN, self.STD, self.test_transforms)
            else:
                self.testgen = SLArrayDataset('test', self.testimages, self.testlabels, self.MEAN, self.STD, self.test_transforms)

    def train_dataloader(self, pretrain : Optional[bool] = True):
        if pretrain:
            trainloader = DataLoader(self.traingen, batch_size = self.pretrain_batch_size, shuffle = True, drop_last = True)
        else:
            trainloader = DataLoader(self.traingen, batch_size = self.other_batch_size, shuffle = True, drop_last = True)
        return trainloader
    def valid_dataloader(self, pretrain : Optional[bool] = True):
        if pretrain:
            validloader = DataLoader(self.validgen, batch_size = self.pretrain_batch_size, drop_last = True)
        else:
            validloader = DataLoader(self.validgen, batch_size = self.other_batch_size, drop_last = True)
        return validloader
    def test_dataloader(self, pretrain : Optional[bool] = True):
        if pretrain:
            testloader = DataLoader(self.testgen, batch_size = self.pretrain_batch_size, drop_last = True)
        else:
            testloader = DataLoader(self.testgen, batch_size = self.other_batch_size, drop_last = True)
        return testloader

class STL10DataModule(nn.Module):
    def __init__(self,
                 pretrain_batch_size: int = 64,
                 other_batch_size: int = 32,
                 download: bool = False,
                 dataset_path: str = None,
                 transformations: Any = None
                 ) -> None:
        super().__init__()
        self.pretrain_batch_size = pretrain_batch_size
        self.other_batch_size = other_batch_size
        self.download = download
        self.dataset_path = dataset_path
        self.transforms = transformations

        # image shape
        self.HEIGHT = 96
        self.WIDTH = 96
        self.DEPTH = 3

        # size of a single image in bytes
        self.SIZE = self.HEIGHT * self.WIDTH * self.DEPTH

        # url of the binary data
        self.DATA_URL = 'http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz'


    @property
    def num_classes(self) -> int:
        return 10

    def read_labels(self, path_to_labels):
        """
        :param path_to_labels: path to the binary file containing labels from the STL-10 dataset
        :return: an array containing the labels
        """
        with open(path_to_labels, 'rb') as f:
            labels = np.fromfile(f, dtype=np.uint8)
            return labels

    def read_all_images(self, path_to_data):
        """
        :param path_to_data: the file containing the binary images from the STL-10 dataset
        :return: an array containing all the images
        """
        with open(path_to_data, 'rb') as f:
            # read whole file in uint8 chunks
            everything = np.fromfile(f, dtype=np.uint8)
            images = np.reshape(everything, (-1, self.DEPTH, self.HEIGHT, self.WIDTH))
            #images = np.transpose(images, (0, 3, 2, 1))
            return images


    def prepare_data(self):
        if self.download:
            url = self.DATA_URL
            r = requests.get(url, stream = True, allow_redirects = True)
            filename = url.split('/')[-1]
            with open(filename,'wb') as f:
                for chunk in tqdm(r.iter_content(chunk_size = 1024)):
                    if chunk:
                        _ = f.write(chunk)

            self.dataset_path = extract_archive('/'.join([os.getcwd(),filename]))

        # path to the binary train file with image data
        self.TRAIN_DATA_PATH = ''.join([self.dataset_path,'/stl10_binary/train_X.bin'])
        # path to the binary train file with labels
        self.TRAIN_LABEL_PATH = ''.join([self.dataset_path,'/stl10_binary/train_y.bin'])
        # path to the binary test file with image data
        self.TEST_DATA_PATH = ''.join([self.dataset_path,'/stl10_binary/test_X.bin'])
        # path to the binary test file with labels
        self.TEST_LABEL_PATH = ''.join([self.dataset_path,'/stl10_binary/test_y.bin'])
        # path to the binary train file with image data
        self.UNLAB_DATA_PATH = ''.join([self.dataset_path,'/stl10_binary/unlabeled.bin'])

        #DATA READING AND SPLITTING
        #-----training data
        #airplane, bird, car, cat, deer, dog, horse, monkey, ship, truck
        self.lab_dict = {0:'airplane',1:'bird',2:'car',3:'cat',4:'deer',5:'dog',6:'horse',7:'monkey',8:'ship',9:'truck'}
        
        self.trimages = self.read_all_images(self.TRAIN_DATA_PATH)
        self.trlabels = self.read_labels(self.TRAIN_LABEL_PATH) - 1 # IN STL10 LABELS RANGE FROM 1 TO 10 SO THE LABELS NEED TO BE BROUGHT TO 0 TO 9
        self.trimages = self.trimages.astype(np.float32)
        self.trimages = self.trimages/255.0

        # splitting training and validation data
        self.trimages, self.valimages, self.trlabels, self.vallabels = train_test_split(self.trimages, self.trlabels, test_size = 0.1)

        self.MEAN = np.mean(self.trimages, axis = (0,2,3), keepdims = False)
        self.STD = np.std(self.trimages, axis = (0,2,3), keepdims = False)
        
        #----TEST IMAGES
        self.testimages = self.read_all_images(self.TEST_DATA_PATH)
        self.testlabels = self.read_labels(self.TEST_LABEL_PATH) - 1 
        self.testimages = self.testimages.astype(np.float32)
        self.testimages = self.testimages/255.0

    def setup(self, stage: Optional[str] = None, pretrain : Optional[bool] = True, fracs: Optional[float] = 1.0):
        #transforms
        if stage == 'train':
            self.train_transforms = self.transforms
            if pretrain:
                self.traingen = SSLArrayDataset('train', self.trimages, self.trlabels, self.MEAN, self.STD, self.train_transforms)
            else:
                self.traingen = SLArrayDataset('train', self.trimages, self.trlabels, self.MEAN, self.STD, transforms.RandomResizedCrop(self.HEIGHT,(0.8,1.0)), fracs)
        if stage == 'valid':
            self.valid_transforms = None #transforms.Compose([transforms.Normalize(self.MEAN, self.STD)])
            if pretrain:
                self.validgen = SSLArrayDataset('valid', self.valimages, self.vallabels, self.MEAN, self.STD, self.valid_transforms) #torchvision.transforms.RandomResizedCrop(32,(0.8,1.0))
            else:
                self.validgen = SLArrayDataset('valid', self.valimages, self.vallabels, self.MEAN, self.STD, self.valid_transforms) #torchvision.transforms.RandomResizedCrop(32,(0.8,1.0))

        if stage == 'test':
            self.test_transforms = None #transforms.Compose([transforms.Normalize(self.MEAN, self.STD)])
            if pretrain:
                self.testgen = SSLArrayDataset('test', self.testimages, self.testlabels, self.MEAN, self.STD, self.test_transforms)
            else:
                self.testgen = SLArrayDataset('test', self.testimages, self.testlabels, self.MEAN, self.STD, self.test_transforms)

    def train_dataloader(self, pretrain : Optional[bool] = True):
        if pretrain:
            trainloader = DataLoader(self.traingen, batch_size = self.pretrain_batch_size, shuffle = True, drop_last = True)
        else:
            trainloader = DataLoader(self.traingen, batch_size = self.other_batch_size, shuffle = True, drop_last = True)
        return trainloader
    def valid_dataloader(self, pretrain : Optional[bool] = True):
        if pretrain:
            validloader = DataLoader(self.validgen, batch_size = self.pretrain_batch_size, drop_last = True)
        else:
            validloader = DataLoader(self.validgen, batch_size = self.other_batch_size, drop_last = True)
        return validloader
    def test_dataloader(self, pretrain : Optional[bool] = True):
        if pretrain:
            testloader = DataLoader(self.testgen, batch_size = self.pretrain_batch_size, drop_last = True)
        else:
            testloader = DataLoader(self.testgen, batch_size = self.other_batch_size, drop_last = True)
        return testloader

