import os
import pathlib
import time
import datetime
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from apex import amp
from torch.utils.data import DataLoader
from utils.core import evaluate
from utils.builder import *
from utils.utils import *
from utils.meter import AverageMeter
from utils.logger import Logger
from coldl import ResNet
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--nclasses', type=int, required=True)
    parser.add_argument('--gpu', type=str)
    args = parser.parse_args()

    init_seeds()
    device = set_device(args.gpu)

    transform = build_transform(rescale_size=448, crop_size=448)
    dataset = build_webfg_dataset(os.path.join('Datasets', args.dataset), transform['train'], transform['test'])
    net = ResNet(args.arch, num_classes=args.nclasses).to(device)
    net.load_state_dict(torch.load(args.model_path))
    test_loader = DataLoader(dataset['test'], batch_size=32, shuffle=False, num_workers=8, pin_memory=True)
    test_accuracy = evaluate(test_loader, net, device)
    
    print(f'Test accuracy: {test_accuracy:.3f}')
