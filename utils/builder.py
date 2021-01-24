import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from data.image_folder import IndexedImageFolder
import numpy as np


def build_transform(rescale_size=512, crop_size=448):
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=rescale_size),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomCrop(size=crop_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=rescale_size),
        torchvision.transforms.CenterCrop(size=crop_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    return {'train': train_transform, 'test': test_transform}

def build_webfg_dataset(root, train_transform, test_transform):
    train_data = IndexedImageFolder(os.path.join(root, 'train'), transform=train_transform)
    test_data = IndexedImageFolder(os.path.join(root, 'val'), transform=test_transform)
    return {'train': train_data, 'test': test_data, 'n_train_samples': len(train_data.samples), 'n_test_samples': len(test_data.samples)}

def build_sgd_optimizer(params, lr, weight_decay, nesterov=True):
    return optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9, nesterov=nesterov)
