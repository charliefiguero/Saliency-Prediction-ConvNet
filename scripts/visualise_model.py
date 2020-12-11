#!/usr/bin/env python3
import math
import time
import sys
import os
from multiprocessing import cpu_count
from typing import Union, NamedTuple

import torch
import torch.backends.cudnn
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
import torchvision.datasets
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import argparse
from pathlib import Path

# from evalutaion.py
import pickle
from PIL import Image
from scipy import ndimage
import dataset
from train_salicon import CNN

from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt

dir_path = "/mnt/storage/home/cf17559/cw"

torch.backends.cudnn.benchmark = True

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

class ImageShape(NamedTuple):
    height: int
    width: int
    channels: int

def main():
    # train_dataset = dataset.Salicon("/mnt/storage/scratch/wp13824/adl-2020/train.pkl")
    test_dataset = dataset.Salicon("/mnt/storage/scratch/wp13824/adl-2020/val.pkl")

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=128,
        num_workers=cpu_count(),
        pin_memory=True,
    )

    model = torch.load('model_13-53-55/model_999.pkl')
    
    visualise(model, test_loader, "repli_vis_0")
    visualise(model, test_loader, "repli_vis_1")
    visualise(model, test_loader, "repli_vis_2")
    visTensor(model.conv1.weight.detach().cpu().clone())

def visTensor(tensor, ch=0, allkernels=False, nrow=8, padding=1): 

    fig, ax = plt.subplots(nrows=6,ncols=6,figsize=(16,12))
    
    n,c,w,h = tensor.shape

    if allkernels: tensor = tensor.view(n*c, -1, w, h)
    elif c != 3: tensor = tensor[:,ch,:,:].unsqueeze(dim=1)

    rows = np.min((tensor.shape[0] // nrow + 1, 64))    
    grid = utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
    plt.figure( figsize=(nrow,rows) )
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.savefig('conv1_filters.png')

def visualise(model, val_loader, fig_name):
    preds = []
    model.eval()

    # No need to track gradients for validation, we're not optimizing.
    with torch.no_grad():
        for batch, labels in val_loader:
            batch = batch.to(DEVICE)
            logits = model(batch)
            preds_output = logits.cpu().numpy()
            preds.extend(list(preds_output))

    with open("/mnt/storage/scratch/wp13824/adl-2020/val.pkl", 'rb') as f:
        gts = pickle.load(f)
        
    index = np.random.randint(0, len(preds), size=3) #get indices for 3 random images

    outputs = []
    for idx in index:
        #getting original image
        image = gts[idx]['X_original']
        image = np.swapaxes(np.swapaxes(image, 0, 1), 1, 2)
        outputs.append(image)

        #getting ground truth saliency map
        sal_map = gts[idx]['y_original']
        sal_map = ndimage.gaussian_filter(sal_map, 19)
        outputs.append(sal_map)

        #getting model prediction
        pred = np.reshape(preds[idx], (48, 48))
        pred = Image.fromarray((pred * 255).astype(np.uint8)).resize((image.shape[1], image.shape[0]))
        pred = np.asarray(pred, dtype='float32') / 255.
        pred = ndimage.gaussian_filter(pred, sigma=2)
        outputs.append(pred)

    #plotting images 
    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(32,32))
    ax[0][0].set_title("Image", fontsize=40)
    ax[0][1].set_title("Ground Truth", fontsize=40)
    ax[0][2].set_title("Prediction", fontsize=40)
    
    fig.tight_layout()

    for i, axi in enumerate(ax.flat):
        axi.imshow(outputs[i])
    
    outpath = os.path.join(dir_path, fig_name+".jpg")
    plt.savefig(outpath)

if __name__ == "__main__":
    main()