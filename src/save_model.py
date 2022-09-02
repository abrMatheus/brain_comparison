from flim.experiments import utils

import numpy as np

import torch

from torch import optim

from torchvision import transforms
import torchio as tio

from torch_snippets import *

from model import UNet, UnetLoss, train_batch, validate_batch, IoU
from data.dataloader import SegmDataset, ToTensor
from data.bratsdataset import BratsDataset, get_train_transforms, get_test_transforms
from torch.utils.data import random_split, DataLoader
import torch as th


from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from typing import Dict, Tuple, Any, Callable
from metrics.metrics import multiclass_dice, multiclass_sensitivity
from models.resunet import ResUNet
from monai.visualize.img2tensorboard import add_animated_gif

import nibabel as nib


from experiment import LitModel
import sys
import os

arch = utils.load_architecture(sys.argv[1])
parampath = sys.argv[2]


encoder = utils.build_model(arch, input_shape=[3])
encoder2 = utils.build_model(arch, input_shape=[3])

encoder = utils.load_weights_from_lids_model(encoder, os.path.join(parampath, 'flair'))

encoder2 = utils.load_weights_from_lids_model(encoder2, os.path.join(parampath, 't1gd'))

unet = UNet(encoder1=encoder, encoder2=encoder2, out_channels=3,train_encoder=False)

model = LitModel(unet, optim='adam', lr=1e-5)

utils.save_lids_model(unet.encoder1, arch, 'output_dir', 'model0/flair')

utils.save_lids_model(unet.encoder2, arch, 'output_dir', 'model0/t1gd')
