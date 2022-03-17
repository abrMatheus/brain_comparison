from flim.experiments import utils

import numpy as np

import torch

from torch import optim

from torchvision import transforms

from torch_snippets import *

from model import UNet, UnetLoss, train_batch, validate_batch, get_device, IoU
from data.dataloader import SegmDataset, ToTensor
import torch as th


from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from typing import Dict, Tuple, Any, Callable
from metrics.metrics import multiclass_dice, multiclass_sensitivity
from monai.visualize.img2tensorboard import add_animated_gif

import nibabel as nib
import sys

from experiment import getInsideModel, LitModel, getDataloaders


def save_predition(y_hat, batch, output_folder, count=0):
    data = y_hat.detach().numpy()[0]
    data = data.transpose(1,2,3,0).astype(np.int16)
    data = data.argmax(axis=3)
    data = data.astype(np.int16)
    affine = np.eye(4)*np.array([-1,-1,1,1])
    imgNib = nib.Nifti1Image(data, affine=affine)

    name = str(count) + '.nii.gz'
    
    output_path = os.path.join(output_folder , name)
    print("saving to ", output_path)
    nib.save(imgNib, output_path)

def run_experiment(datapath='/app/data', batchsize=1, archpath='/app/arch.json',
                   parampath='brain3d-param-small', n_epochs=1, exp_name='test',
                   checkpoint_file='model.ckpt',modeltype='flimunet', datatype='ours'):
    
    insidemodel = getInsideModel(modeltype, out_channels=4, archpath=archpath, parampath=parampath)

    model = LitModel(insidemodel, optim='adam', lr=2e-4)

    transform = transforms.Compose([ToTensor()])
    trn_dl, val_dl, test_dl = getDataloaders(datatype, datapath, transform, batchsize, modeltype, num_workers=8)

    model_checkpoint = ModelCheckpoint(
        monitor='val_loss',
        dirpath='exp/',
        filename=exp_name + '{epoch:02d}-{val_loss:.2f}-{val_WT_dice:.2f}_val',
        save_top_k=3,
        mode='min',
    )

    logger = TensorBoardLogger('exp/logs', name=exp_name)
    trainer = pl.Trainer(
        gpus=[2],
        #accelerator='ddp',
        precision=16,
        accumulate_grad_batches=1,#accum_batch_size,
        terminate_on_nan=True,
        max_epochs=n_epochs,
        logger=logger,
        callbacks=[LearningRateMonitor(logging_interval='step'), model_checkpoint],
    )


    print('file', checkpoint_file)
    checkpoint = th.load(checkpoint_file)
    print("state_dict keys")
    model.load_state_dict(checkpoint['state_dict'])


    # training
    #trainer.fit(model, trn_dl, val_dl)


    trainer.test(model, test_dl)

    for step, batch in enumerate(val_dl):
        xf, xt, gt = batch[0], batch[1], batch[2]
        model.eval()
        y_hat = model.forward(xf, xt)
        save_predition(y_hat, batch, './out', step)


if __name__ == '__main__':

    file_ckpt = sys.argv[1]

    #run_experiment(datapath='/dados/matheus/dados/glioblastoma/perc/50',batchsize=1, 
    #               archpath='/dados/matheus/git/u-net-with-flim2/archift3d.json',
    #               parampath='/dados/matheus/git/u-net-with-flim2/brain3d-large-param',
    #               checkpoint_file=file_ckpt)

    run_experiment(datapath='/app/glioblastoma/won4/100',batchsize=1,
                   archpath='/app/archs/small/arch.json',
                   parampath='/app/params/roi-won4-small-std',
                   datatype='ours', modeltype='flimunet',
                   checkpoint_file=file_ckpt)


    #run_experiment(datapath='/app/glioblastoma/won4/100',batchsize=1,
    #               archpath='/app/archs/small/arch.json',
    #               parampath='/app/params/roi-won4-small-std',
    #               datatype='ours', modeltype='flimunet',
    #               n_epochs=int(epochs), exp_name='won4_100')

    #run_experiment(datapath='/app/glioblastoma/iqr/100',batchsize=1,
    #               archpath='/app/archs/small/arch.json',
    #               parampath='/app/params/roi-iqr-small',
    #               datatype='ours', modeltype='flimunet',
    #               n_epochs=int(epochs), exp_name='iqr_100')
