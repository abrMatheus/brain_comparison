from flim.experiments import utils

import numpy as np

import torch

from torch import optim

from torchvision import transforms

from torch_snippets import *

from model import UNet, UnetLoss, train_batch, validate_batch, get_device, IoU
from dataloader import SegmDataset, ToTensor
import torch as th


from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from typing import Dict, Tuple, Any, Callable
from metrics.metrics import multiclass_dice, multiclass_sensitivity
from monai.visualize.img2tensorboard import add_animated_gif

import nibabel as nib

class LitModel(pl.LightningModule):
    colormap = th.tensor([[0, 0, 0],
                          [64, 64, 64],
                          [128, 128, 128],
                          [255, 255, 255]], dtype=th.uint8)

    def __init__(self, model: th.nn.Module, optim: str, lr: float) -> None:
        super().__init__()
        self.model = model
        self.optim = optim
        self.lr = lr

        self.class_weights = th.tensor([0.1, 1, 1, 1])
    
    def configure_optimizers(self):#necessario????
        if self.optim.lower() == 'adam':
            optim = th.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optim.lower() == 'sgd':
            optim = th.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
        else:
            raise NotImplementedError
        scheduler = th.optim.lr_scheduler.MultiStepLR(optim, milestones=[35, 45], gamma=0.25)
        return {
            'optimizer': optim,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss',
        }

    @staticmethod
    #necessario??
    def _maybe_resize(x: th.Tensor, shape: Tuple[int]) -> th.Tensor:
        if x.shape[-3:] != shape[-3:]:
            x = F.interpolate(x, size=shape[-3:],
                            mode="trilinear", align_corners=True)
        return x

    @staticmethod
    #necessario??
    def _get_2D_slice(x: th.Tensor) -> th.Tensor:
        """ iteratively slice the array in the middle until it is 2D"""
        x = x.cpu()
        while x.ndim != 3:
            x = x[0]
        x = x[:, :, x.shape[2] // 2]
        return x
    
    def _color_mask(self, x: th.Tensor) -> th.Tensor:
        assert x.ndim == 3
        return self.colormap[x].permute((3, 0, 1, 2))

    def _add_gif(self, x: th.Tensor, tag: str) -> None:
        add_animated_gif(
            writer=self.logger.experiment,
            tag=tag,
            image_tensor=x.cpu(),
            max_out=1,
            scale_factor=1 if x.shape[0] == 3 else 255,
            global_step=self.global_step
        )
        
    def forward(self, xf: th.Tensor, xt1: th.Tensor) -> th.Tensor:
        return self.model.forward(xf, xt1)

    def _step(self, batch: Tuple[th.Tensor], mode: str) -> Any:
        y   = batch[2]
        xt1 = batch[1]
        xf  = batch[0]

        y_hat = self.forward(xf, xt1)
        y_hat = self._maybe_resize(y_hat, y.shape)
        loss, acc = UnetLoss(y_hat, y)
        
        self.log(f'{mode}_loss', loss, on_epoch=True, on_step=True)
        if mode == 'train':
            return loss
        elif mode == 'val' or mode == 'test':
            logits = th.softmax(y_hat, dim=1)
            dice   = multiclass_dice(logits, y)
            prec   = multiclass_sensitivity(logits, y)
            metrics = {
                f'{mode}_WT_dice': dice[1],
                f'{mode}_TC_dice': dice[2],
                f'{mode}_ET_dice': dice[3],

                f'{mode}_WT_sens': prec[1],
                f'{mode}_TC_sens': prec[2],
                f'{mode}_ET_sens': prec[3],
            }
            self.log_dict(metrics)
            if self.global_rank == 0:
                preds = logits.argmax(dim=1)
                self._add_gif(xf[0, :1], f'{mode}/flair')
                self._add_gif(xt1[0, :1], f'{mode}/t1ce')
                self._add_gif(self._color_mask(y[0,0]), f'{mode}/gt')
                self._add_gif(self._color_mask(preds[0]), f'{mode}/pred')
            
            return metrics

        else:
            raise NotImplementedError
    
    def training_step(self, batch: Tuple[th.Tensor], batch_idx: int) -> th.Tensor:
        return self._step(batch, 'train')

    def validation_step(self, batch: Tuple[th.Tensor], batch_idx: int) -> Dict:
        return self._step(batch, mode='val')

    def test_step(self, batch: Tuple[th.Tensor], batch_idx: int) -> th.Tensor:
        return self._step(batch, mode='test')


def save_predition(y_hat, batch, output_folder, count=0):
    data = y_hat.detach().numpy()[0]
    data = data.transpose(1,2,3,0).astype(np.int16)
    imgNib = nib.Nifti1Image(data, affine=np.eye(4))

    name = str(count) + '.nii.gz'
    
    output_path = os.path.join(output_folder , name)
    print("saving to ", output_path)
    nib.save(imgNib, output_path)

def run_experiment(datapath='/app/data', batchsize=1, archpath='/app/arch.json',
                   parampath='brain3d-param-small', n_epochs=30, exp_name='test'):

    device = get_device()

    arch = utils.load_architecture(archpath)

    #input_shape = [H, W, C] or [C]
    encoder = utils.build_model(arch, input_shape=[3])
    encoder = utils.load_weights_from_lids_model(encoder, parampath+ "/flair")


    encoder2 = utils.build_model(arch, input_shape=[3])
    encoder2 = utils.load_weights_from_lids_model(encoder2,parampath+ "/t1gd")


    num_classes = 4
    u_net = UNet(encoder1=encoder, encoder2=encoder2, out_channels=num_classes)

    model = LitModel(u_net, optim='adam', lr=1e-3)

    transform = transforms.Compose([ToTensor()])

    trn_ds  = SegmDataset(datapath, transform=transform, train=True, gts=True)
    val_ds  = SegmDataset(datapath, transform=transform, train=False, gts=True)
    test_ds = SegmDataset(datapath, transform=transform, train=False, gts=True, test=True)

    trn_dl  = DataLoader(trn_ds, batch_size=batchsize, num_workers=8)
    val_dl  = DataLoader(val_ds, batch_size=batchsize, num_workers=8)
    test_dl = DataLoader(test_ds, batch_size=batchsize, num_workers=8)
    
    model_checkpoint = ModelCheckpoint(
        monitor='val_WT_dice',
        dirpath='exp/',
        filename=exp_name + '{epoch:02d}-{val_loss:.2f}-{val_WT_dice:.2f}',
        save_top_k=1,
        mode='max',
    )

    logger = TensorBoardLogger('logs', name=exp_name)
    trainer = pl.Trainer(
        gpus=1,
        #accelerator='ddp',
        precision=16,
        accumulate_grad_batches=1,#accum_batch_size,
        terminate_on_nan=True,
        max_epochs=n_epochs,
        logger=logger,
        callbacks=[LearningRateMonitor(logging_interval='step'), model_checkpoint],
    )
    # training
    trainer.fit(model, trn_dl, val_dl)

    trainer.test(model, test_dl, ckpt_path=model_checkpoint.best_model_path)

if __name__ == '__main__':

    exp_name = sys.argv[1]
    epochs   = sys.argv[2]

    run_experiment(datapath='/dados/matheus/dados/glioblastoma/perc/50',batchsize=1, 
                   archpath='/dados/matheus/git/u-net-with-flim2/archift3d-small.json',
                   parampath='/dados/matheus/git/u-net-with-flim2/brain3d-small-param',
                   n_epochs=int(epochs), exp_name=exp_name)
