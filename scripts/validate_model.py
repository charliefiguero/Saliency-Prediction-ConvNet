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

# importing dataset which will unpickle the SALICON data, ready to be passed to a torch Dataloader.
sys.path.append("/Users/charlesfiguero/Documents/Saliency-Prediction-ConvNet/ADL CW")
import dataset
import evaluation

from train_salicon_plus import CNN

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
    try:
        # train_dataset = dataset.Salicon("/mnt/storage/scratch/wp13824/adl-2020/train.pkl")
        test_dataset = dataset.Salicon("/mnt/storage/scratch/wp13824/adl-2020/val.pkl")
    except:
        # train_dataset = dataset.Salicon("../ADL CW/train.pkl")
        test_dataset = dataset.Salicon("../ADL CW/val.pkl")

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=128,
        num_workers=cpu_count(),
        pin_memory=True,
    )

    model = torch.load('models/model1.pth')

    validate(model, test_loader)

def validate(model, val_loader):
    results = {"preds": [], "labels": []}
    total_loss = 0
    model.eval()

    # No need to track gradients for validation, we're not optimizing.
    with torch.no_grad():
        for batch, labels in val_loader:

            batch = batch.to(DEVICE)
            labels = labels.to(DEVICE)
            logits = model(batch)
            preds = logits.cpu().numpy()
            results["preds"].extend(list(preds))

    with open("/mnt/storage/scratch/wp13824/adl-2020/val.pkl", 'rb') as f:
        val = pickle.load(f)

    preds = results["preds"]
    cc_scores = []
    auc_borji_scores = []
    auc_shuffled_scores = []
    for i in range(len(preds)):

        gt = val[i]['y_original']
        pred = np.reshape(preds[i], (48, 48))
        pred = Image.fromarray((pred * 255).astype(np.uint8)).resize((gt.shape[1], gt.shape[0]))
        pred = np.asarray(pred, dtype='float32') / 255.
        pred = ndimage.gaussian_filter(pred, sigma=2)
        # pred_blur = ndimage.gaussian_filter(pred, sigma=19)

        #cc
        cc_scores.append(evaluation.cc(pred, gt))

        # borji
        auc_borji_scores.append(evaluation.auc_borji(pred, np.asarray(gt, dtype=np.int)))

        # auc-shuffle
        other = np.zeros((gt.shape[0], gt.shape[1]), dtype=np.int)
        randind_maps = np.random.choice(len(val), size=10, replace=False)
        for i in range(10):
            other = other | np.asarray(val[randind_maps[i]]['y_original'], dtype=np.int)

        auc_shuffled_scores.append(evaluation.auc_shuff(pred, np.asarray(gt, dtype=np.int), other))

    accuracy = np.mean(cc_scores)
    accuracy_borji = np.mean(auc_borji_scores)
    accuracy_auc_shuffled = np.mean(auc_shuffled_scores)

    print(f"cross correlation accuracy: {accuracy * 100:2.2f}")
    print(f"auc borji accuracy: {accuracy_borji * 100:2.2f}")
    print(f"auc shuffle accuracy: {accuracy_auc_shuffled * 100:2.2f}")

if __name__ == '__main__':
    main()