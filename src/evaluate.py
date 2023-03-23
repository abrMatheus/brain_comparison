from flim.experiments import utils

import numpy as np

import torch

from torch import optim

from torchvision import transforms

from torch_snippets import *

from model import UNet, UnetLoss, train_batch, validate_batch, get_device, IoU
from datas.dataloader import SegmDataset, ToTensor
from datas.bratsdataset import BratsDataset, get_train_transforms, get_test_transforms
import torch as th


from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from typing import Dict, Tuple, Any, Callable
from metrics.metrics import multiclass_dice, multiclass_sensitivity
from monai.visualize.img2tensorboard import add_animated_gif

import nibabel as nib
import sys

from experiment import getInsideModel, LitModel, getDataloaders



def maybe_resize(x: th.Tensor, shape: Tuple[int]) -> th.Tensor:
        if x.shape[-3:] != shape[-3:]:
            x = F.interpolate(x, size=shape[-3:],
                            mode="trilinear", align_corners=True)
        return x



#save_pred_brats(y_hat, batch, './out', batch['name'])
def save_pred_brats(y_hat, batch, output_folder, im_name):
    data = maybe_resize(y_hat, batch['flair'].shape)

    data = data.detach().numpy()[0]
    data = data.transpose(3,2,1,0).astype(np.int16)
    data = data.argmax(axis=3)
    data = data.astype(np.int16)
    data[data==3]=4


    print(data.shape, batch['flair'].shape)

    affine = np.eye(4)*np.array([-1,-1,1,1])
    imgNib = nib.Nifti1Image(data, affine=affine)

    name = str(im_name) + '_seg.nii.gz'

    output_path = os.path.join(output_folder , name)
    print("saving to ", output_path)
    nib.save(imgNib, output_path)

def save_predition(y_hat, batch, output_folder, im_name):
    
    data = maybe_resize(y_hat, batch[1].shape)


    data = data.detach().numpy()[0]
    data = data.transpose(3,2,1,0).astype(np.int16)
    data = data.argmax(axis=3)
    data = data.astype(np.int16)
    
    print(data.shape, batch[1].shape)

    affine = np.eye(4)*np.array([-1,-1,1,1])
    imgNib = nib.Nifti1Image(data, affine=affine)

    name = str(im_name) + '.nii.gz'
    
    output_path = os.path.join(output_folder , name)
    print("saving to ", output_path)
    nib.save(imgNib, output_path)


def getDataloaders_brats(datapath, transform, batch_size, model, num_workers=5):

    dataset = BratsDataset( datapath, mode='train', model=model)
    test_dl = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, persistent_workers=True)
    return test_dl

def run_experiment(datapath='/app/data', batchsize=1, archpath='/app/arch.json',
                   parampath='brain3d-param-small', n_epochs=1, exp_name='test',
                   checkpoint_file='model.ckpt',modeltype='flimunet', datatype='ours', use_bias=False):
    
    insidemodel,arch = getInsideModel(modeltype, out_channels=4, archpath=archpath, parampath=parampath, use_bias=use_bias)

    model = LitModel(insidemodel, optim='adam', lr=2e-4)

    transform = transforms.Compose([ToTensor()])
    if datatype=='ours':
        trn_dl, val_dl, test_dl = getDataloaders(datatype, datapath, transform, batchsize, modeltype, num_workers=8)
    elif datatype=='brats':
        test_dl = getDataloaders_brats(datapath, transform, batchsize, modeltype, num_workers=8)

    model_checkpoint = ModelCheckpoint(
        monitor='val_loss',
        dirpath='exp/',
        filename=exp_name + '{epoch:02d}-{val_loss:.2f}-{val_WT_dice:.2f}_val',
        save_top_k=3,
        mode='min',
    )

    logger = TensorBoardLogger('exp/logs', name=exp_name)
    trainer = pl.Trainer(
        gpus=[1],
        #accelerator='ddp',
        precision=16,
        accumulate_grad_batches=1,#accum_batch_size,
        max_epochs=n_epochs,
        logger=logger,
        callbacks=[LearningRateMonitor(logging_interval='step'), model_checkpoint],
    )


    print('file', checkpoint_file)
    checkpoint = th.load(checkpoint_file)
    #print("state_dict keys", checkpoint['state_dict'].keys())
    model.load_state_dict(checkpoint['state_dict'])

    model=model.cuda(1)


    if datatype == 'ours':

        for step, batch in enumerate(test_dl):
            xf, xt, gt = batch[0], batch[1], batch[2]
            model.eval()
            #print('min max', xf.min(), xf.max(), xt.min(), xt.max())
            #print('shape', xf.shape, xt.shape)
            y_hat = model.forward(xf.cuda(1), xt.cuda(1))
            save_predition(y_hat.detach().cpu(), batch, './out', batch[3][0])

    elif datatype == 'brats':
        for step, batch in enumerate(test_dl):
           #print(type(batch))
           #print(batch.keys())
           #print('name', batch['name'])
           xf,xt = batch['flair'], batch['t1ce']
           #print('min max', xf.min(), xf.max(), xt.min(), xt.max())
           #print('shape', xf.shape, xt.shape)
           #break
           y_hat = model.forward(xf.cuda(1),xt.cuda(1))
           save_pred_brats(y_hat.detach().cpu(), batch, './out', batch['name'][0])

    #utils.save_lids_model(insidemodel, arch, 'output_models', 'flim_ft')


if __name__ == '__main__':

    file_ckpt='exp/FLIM_adj_FIX_bothLoss_e-4_b1_1gpu_poly_final.ckpt'
    run_experiment(
            datapath='/app/data/glioblastoma/rigid/100',batchsize=1,
            archpath='/app/data/archs/new_small/arch.json',
            parampath='/app/data/ms_files/bkp_models/2enc/new_model_old_t1_adjusted/',
            datatype='ours', modeltype='flimunet',use_bias=False,
            checkpoint_file=file_ckpt)

    file_ckpt='exp/biased_bias_adj_FIX_bothLoss_e-4_b1_1gpu_poly_final.ckpt'
    run_experiment(
            datapath='/app/data/glioblastoma/rigid/100',batchsize=1,
            archpath='/app/data/archs/new_small/arch.json',
            parampath='/app/data/ms_files/bkp_models/2enc_bias/biased_model_2encoders',
            datatype='ours', modeltype='flimunet',use_bias=True,
            checkpoint_file=file_ckpt)
    

    ## running for the brats and using non bias model
    file_ckpt='exp/biased_adj_FIX_bothLoss_e-4_b1_1gpu_poly_final.ckpt'
    run_experiment(
            #datapath='/app/data/glioblastoma/rigid/100',batchsize=1,
            datapath='/app/data/', batchsize=1, #for the brats we do not pass the whole path
            archpath='/app/data/archs/new_small/arch.json',
            parampath='/app/data/ms_files/bkp_models/2enc_bias/biased_model_2encoders',
            datatype='brats', modeltype='flimunet',use_bias=False,
            checkpoint_file=file_ckpt)