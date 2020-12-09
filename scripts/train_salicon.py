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

from datetime import datetime

# importing dataset which will unpickle the SALICON data, ready to be passed to a torch Dataloader.
# path needs to change
import dataset
import evaluation

now = datetime.now()
current_time = now.strftime("%H-%M-%S")
model_dir_path = "/mnt/storage/home/cf17559/cw/model_"+current_time
print("Current Time =", current_time)

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(
    description="Train a shallow saliency prediction covnet on the SALICON dataset.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument("--log-dir", default=Path("logs"), type=Path)
parser.add_argument("--learning-rate", default=0.01, type=float, help="Learning rate")
parser.add_argument(
    "--batch-size",
    default=128,
    type=int,
    help="Number of images within each mini-batch",
)
parser.add_argument(
    "--epochs",
    default=100,
    type=int,
    help="Number of epochs (passes through the entire dataset) to train for",
)
parser.add_argument(
    "--val-frequency",
    default=2,
    type=int,
    help="How frequently to test the model on the validation set in number of epochs",
)
parser.add_argument(
    "--log-frequency",
    default=10,
    type=int,
    help="How frequently to save logs to tensorboard in number of steps",
)
parser.add_argument(
    "--print-frequency",
    default=10,
    type=int,
    help="How frequently to print progress to the command line in number of steps",
)
parser.add_argument(
    "-j",
    "--worker-count",
    default=cpu_count(),
    type=int,
    help="Number of worker processes used to load data.",
)

class ImageShape(NamedTuple):
    height: int
    width: int
    channels: int

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

def normalize_map(s_map):
    norm_s_map = (s_map - np.min(s_map)) / (np.max(s_map) - np.min(s_map))
    return norm_s_map

def main(args):
    os.mkdir(model_dir_path)

    train_dataset = dataset.Salicon("/mnt/storage/scratch/wp13824/adl-2020/train.pkl")
    test_dataset = dataset.Salicon("/mnt/storage/scratch/wp13824/adl-2020/val.pkl")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.worker_count,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.worker_count,
        pin_memory=True,
    )

    model = CNN(height=96, width=96, channels=3)
    
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)

    log_dir = get_summary_writer_log_dir(args)
    print(f"Writing logs to {log_dir}")
    summary_writer = SummaryWriter(
            str(log_dir),
            flush_secs=5
    )
    trainer = Trainer(
        model, train_loader, test_loader, criterion, summary_writer, DEVICE
    )

    trainer.train(
        args.epochs,
        args.val_frequency,
        print_frequency=args.print_frequency,
        log_frequency=args.log_frequency,
    )

    summary_writer.close()

class CNN(nn.Module):
    def __init__(self, 
                 height: int = 96,
                 width: int = 96,
                 channels: int = 3):
        super().__init__()
        self.input_shape = ImageShape(height=height, width=width, channels=channels)

        self.conv1 = nn.Conv2d(
            in_channels=self.input_shape.channels,
            out_channels=32,
            kernel_size=(5, 5),
            padding=2,
        )
        self.initialise_layer(self.conv1)

        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=(3, 3),
            padding=1,
        )
        self.initialise_layer(self.conv2)

        self.pool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=2)

        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=(3, 3),
            padding=1,
        )
        self.initialise_layer(self.conv3)

        self.pool3 = nn.MaxPool2d(kernel_size=(3, 3), stride=2)

        self.fc1 = nn.Linear(11*11*128, 4608) # check why we get 11x11 and paper gets 10x10
        self.initialise_layer(self.fc1)

        self.fc2 = nn.Linear(2304, 2304)
        self.initialise_layer(self.fc2)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.conv1(images)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool3(x)

        # Flatten the output of the pooling layer so it is of shape (batch_size, 4096)    
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)

        # MAXOUT
        x = x.reshape(x.shape[0], 2304, 2)
        x = torch.max(x, 2)[0]
        x = F.relu(x)

        x = self.fc2(x)
        return x

    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            layer.bias.data.fill_(0.1)
        if hasattr(layer, "weight"):
            layer.weight.data.normal_(0.0, 0.01)

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        summary_writer: SummaryWriter,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = torch.optim.SGD(self.model.parameters(), nesterov=True, momentum=0.9, lr=0.03, weight_decay=0.0005)
        self.summary_writer = summary_writer
        self.step = 0

    def train(
        self,
        epochs: int,
        val_frequency: int,
        print_frequency: int = 20,
        log_frequency: int = 5,
        start_epoch: int = 0
    ):
        learning_rates = np.linspace(0.03, 0.0001,epochs)
        for epoch in range(start_epoch, epochs):
            self.model.train()
            data_load_start_time = time.time()
            for batch, labels in self.train_loader:
                
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = learning_rates[epoch]

                batch = batch.to(self.device)
                labels = labels.to(self.device)
                data_load_end_time = time.time()

                self.optimizer.zero_grad()
                logits = self.model.forward(batch)

                loss = self.criterion(logits, labels)

                loss.backward()

                self.optimizer.step()

                data_load_time = data_load_end_time - data_load_start_time
                step_time = time.time() - data_load_end_time
                if ((self.step + 1) % log_frequency) == 0:
                    self.log_metrics(epoch, loss, data_load_time, step_time)
                    # self.log_metrics(epoch, accuracy, loss, data_load_time, step_time)
                if ((self.step + 1) % print_frequency) == 0:
                    self.print_metrics(epoch, loss, data_load_time, step_time)
                    # self.print_metrics(epoch, accuracy, loss, data_load_time, step_time)

                self.step += 1
                data_load_start_time = time.time()

            self.summary_writer.add_scalar("epoch", epoch, self.step)
            if ((epoch + 1) % 2) == 0:
                self.validate()
                # self.validate() will put the model in validation mode,
                # so we have to switch back to train mode afterwards
                self.model.train()
            if ((epoch + 1) % val_frequency) == 0:           
                torch.save(self.model, model_dir_path+"/model_"+str(epoch)+".pkl")

                # self.validate() will put the model in validation mode,
                # so we have to switch back to train mode afterwards
                self.model.train()

    def print_metrics(self, epoch, loss, data_load_time, step_time):
        epoch_step = self.step % len(self.train_loader)
        print(
                f"epoch: [{epoch}], "
                f"step: [{epoch_step}/{len(self.train_loader)}], "
                f"batch loss: {loss:.5f}, "
                f"data load time: "
                f"{data_load_time:.5f}, "
                f"step time: {step_time:.5f}"
        )

    def log_metrics(self, epoch, loss, data_load_time, step_time):
        self.summary_writer.add_scalar("epoch", epoch, self.step)
        self.summary_writer.add_scalars(
                "loss",
                {"train": float(loss.item())},
                self.step
        )
        self.summary_writer.add_scalar(
                "time/data", data_load_time, self.step
        )
        self.summary_writer.add_scalar(
                "time/data", step_time, self.step
        )

    def validate(self):
        results = {"preds": [], "labels": []}
        total_loss = 0
        self.model.eval()

        # No need to track gradients for validation, we're not optimizing.
        with torch.no_grad():
            for batch, labels in self.val_loader:

                batch = batch.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(batch)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()

        average_loss = total_loss / len(self.val_loader)

        self.summary_writer.add_scalars(
                "loss",
                {"test": average_loss},
                self.step
        )

def get_summary_writer_log_dir(args: argparse.Namespace) -> str:
    """Get a unique directory that hasn't been logged to before for use with a TB
    SummaryWriter.

    Args:
        args: CLI Arguments

    Returns:
        Subdirectory of log_dir with unique subdirectory name to prevent multiple runs
        from getting logged to the same TB log directory (which you can't easily
        untangle in TB).
    """
    tb_log_dir_prefix = f'CNN_bs={args.batch_size}_lr={args.learning_rate}_run_'
    i = 0
    while i < 1000:
        tb_log_dir = args.log_dir / (tb_log_dir_prefix + str(i))
        if not tb_log_dir.exists():
            return str(tb_log_dir)
        i += 1
    return str(tb_log_dir)

if __name__ == "__main__":
    main(parser.parse_args())
