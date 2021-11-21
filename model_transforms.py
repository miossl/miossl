from typing import Any
from torchvision import transforms
import torch
from PIL import ImageFilter
import random


class MIOTransform:
    def __init__(self, s, l):
        self.s = s
        self.l = l
        if self.l <= 32:
            self.transforms = transforms.Compose([transforms.RandomHorizontalFlip(0.5),
                            transforms.RandomResizedCrop(self.l,(0.8,1.0)),
                            transforms.Compose([transforms.RandomApply([transforms.ColorJitter(0.8*self.s,
                                                                                                0.8*self.s,
                                                                                                0.8*self.s,
                                                                                                0.2*self.s)], p = 0.8),
                                                                        transforms.RandomGrayscale(p=0.2)])])
        else:
            self.transforms = transforms.Compose([transforms.RandomHorizontalFlip(0.5),
                            transforms.RandomResizedCrop(self.l,(0.8,1.0)),
                            transforms.Compose([transforms.RandomApply([transforms.ColorJitter(0.8*self.s,
                                                                                                0.8*self.s,
                                                                                                0.8*self.s,
                                                                                                0.2*self.s)], p = 0.8),
                                                                        transforms.RandomGrayscale(p=0.2)]),
                            transforms.RandomApply([transforms.GaussianBlur(kernel_size = int(l//10) if int(l//10)%2!=0 else int(l//10)+1, 
                                                                            sigma=(.1, 2.))],
                                                   p = 0.5),
                            transforms.RandomSolarize(threshold = 0.5, p = 0.2)])
    def __call__(self,x):
        x1 = self.transforms(x)
        x2 = self.transforms(x)
        return x1, x2

