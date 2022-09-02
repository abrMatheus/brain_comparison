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

arch = utils.load_architecture('/app/data/archs/new_small/arch.json')

encoder = utils.build_model(arch, input_shape=[3])
encoder2 = utils.build_model(arch, input_shape=[3])

unet = UNet(encoder1=encoder, encoder2=encoder2, out_channels=3,train_encoder=False)

model = LitModel(unet, optim='adam', lr=1e-5)


checkpoint = th.load("exp/rigid_flimepoch=00-val_loss=0.71-val_WT_dice=0.47_dice.ckpt")
#print(checkpoint['state_dict'].keys())

#model.load_state_dict(checkpoint['state_dict'])

utils.save_lids_model(unet.encoder1, arch, 'output_dir', 'model')
#print(model)

#checkpoint


