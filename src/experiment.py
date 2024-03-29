from flim.experiments import utils

import numpy as np

import torch

from torch import optim

from torchvision import transforms
import torchio as tio

from torch_snippets import *

from model import UNet,UNet1enc, UnetLoss, train_batch, validate_batch, IoU
from datas.dataloader import SegmDataset, ToTensor
from datas.bratsdataset import BratsDataset, get_train_transforms, get_test_transforms
from torch.utils.data import random_split, DataLoader
import torch as th


from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from typing import Dict, Tuple, Any, Callable
from metrics.metrics import multiclass_dice, multiclass_sensitivity
from models.resunet import ResUNet
from models.unet import StandUNet
from monai.visualize.img2tensorboard import add_animated_gif

import nibabel as nib

class LitModel(pl.LightningModule):
    colormap = th.tensor([[0, 0, 0],
                          [64, 64, 64],
                          [128, 128, 128],
                          [255, 255, 255]], dtype=th.uint8)

    def __init__(self, model: th.nn.Module, optim: str, lr: float, use_loss='CE', batch_size=1) -> None:
        super().__init__()
        self.model = model
        self.optim = optim
        self.lr = lr
        self.use_loss=use_loss
        self.batch_size=batch_size

        self.class_weights = th.tensor([1, 1, 1, 1])
        self.max_e=100

    def poly_decay(self,i_epoch):
        return (1-(i_epoch/self.max_e))**0.9
    
    def configure_optimizers(self):#necessario????
        if self.optim.lower() == 'adam':
            optim = th.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optim.lower() == 'sgd':
            optim = th.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
        else:
            raise NotImplementedError
        #scheduler = th.optim.lr_scheduler.MultiStepLR(optim, milestones=[35, 45], gamma=0.9)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=self.poly_decay)
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
        #print(f"testing2 {self.model.training} {self.model.encoder1.training} {self.model.encoder2.training}")
        return self.model.forward(xf, xt1)

    def _step(self, batch: Tuple[th.Tensor], mode: str) -> Any:
        #TODO: melhorar isso!!!!
        if isinstance(batch, dict):
            y = batch['seg']#[tio.DATA].long()#.squeeze_(1).long()
            xt1 = batch['t1ce']
            xf = batch['flair']


            #print("sizes", xf.shape, xt1.shape, y.shape)
            
        else:
            y   = batch[2]
            xt1 = batch[1]
            xf  = batch[0]

            #print("sizes", xf.shape, xt1.shape, y.shape)

        
        y_hat = self.forward(xf, xt1)
        y_hat = self._maybe_resize(y_hat, y.shape)
        #print("shapes", y_hat.shape, y.shape)
        loss, acc = UnetLoss(y_hat, y, self.use_loss)
        
        #uncomment for regularization
        #l2_lambda = 0.001
        #l2_norm = sum(p.pow(2.0).sum() for p in self.model.parameters())
        #loss = loss + l2_lambda * l2_norm

        self.log(f'{mode}_loss', loss, on_epoch=True, on_step=True, batch_size=self.batch_size)
        if mode == 'train':
            return {"loss": loss}
        elif mode == 'val' or mode == 'test':
            logits = th.softmax(y_hat, dim=1)
            dice   = multiclass_dice(logits, y)
            prec   = multiclass_sensitivity(logits, y)
            metrics = {
                f'{mode}_WT_dice': dice[1],
                f'{mode}_TC_dice': dice[2],
                f'{mode}_ET_dice': dice[3],
                f'{mode}_ALL_dice': dice[1]*dice[2]*dice[3],

                f'{mode}_WT_sens': prec[1],
                #print("uniques ", np.unique(images[key]))
                f'{mode}_TC_sens': prec[2],
                f'{mode}_ET_sens': prec[3],
                f'{mode}_ALL_sens': prec[1]*prec[2]*prec[3],
            }
            self.log_dict(metrics, batch_size=self.batch_size)
            if self.global_rank == 0:
                preds = logits.argmax(dim=1)
                self._add_gif(xf[0, :1], f'{mode}/flair')
                self._add_gif(xt1[0, :1], f'{mode}/t1ce')
                self._add_gif(self._color_mask(y[0]), f'{mode}/gt')
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
    data = y_hat.detach().numpy()
    data = data.transpose(1,2,3,0)
    data = data.argmax(axis=3)
    data = data.astype(np.int16)
    affine = np.eye(4)*np.array([-1,-1,1,1])
    imgNib = nib.Nifti1Image(data, affine=affine)

    name = str(count) + '.nii.gz'
    
    output_path = os.path.join(output_folder , name)
    print("saving to ", output_path)
    nib.save(imgNib, output_path)


def getDataloaders(datatype, datapath, transform, batch_size, model, num_workers=5, oneEnc=False):

    if datatype == 'brats':
        dataset = BratsDataset( datapath, mode='train', model=model)
        chunk_len = int(len(dataset) * 0.15)
        train_ds, val_ds, test_ds = random_split(dataset,
                                        [len(dataset) - 2 * chunk_len, chunk_len, chunk_len],
                                        generator=th.Generator().manual_seed(42))

        train_ds.dataset.set_transform(get_train_transforms('aug'))
        val_ds.dataset.set_transform(get_test_transforms('aug'))
        test_ds.dataset.set_transform(get_test_transforms('aug'))

        trn_dl = DataLoader(train_ds, batch_size=batch_size,
                                shuffle=True, num_workers=num_workers,
                                persistent_workers=True)

        val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, persistent_workers=True)
        test_dl = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers, persistent_workers=True)


    elif datatype == 'ours':
        trn_ds  = SegmDataset(datapath, transform=transform, train=True, gts=True, model=model,oneEnc=oneEnc)
        val_ds  = SegmDataset(datapath, transform=transform, train=False, gts=True, model=model,oneEnc=oneEnc)
        test_ds = SegmDataset(datapath, transform=transform, train=False, gts=True, test=True, model=model,oneEnc=oneEnc)

        trn_dl  = DataLoader(trn_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_dl  = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers)
        test_dl = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers)

    else:
        raise NotImplementedError(f'there is no {datatype}')

    return trn_dl, val_dl, test_dl


def load_bias_file(path):
    f = open(path, "r")
    N = f.readline()
    N = int(N[:-1])
    bias = f.readline()
    #TODO convert do list and deal with it

    bias = bias.split(" ")[:-1]


    new_bias = []
    for b in bias:
        new_bias.append(float(b))


    return new_bias

def load_first_layer_bias(encoder, model_path):

    bias = load_bias_file(model_path + "/conv1-bias.txt")

    print("bias len", len(bias))

    encoder.features.conv1.bias = torch.nn.Parameter(torch.Tensor(bias))

    return encoder


def getInsideModel(model='resunet', out_channels=4, archpath=None, parampath=None, network_depth=64,
                   train_encoder=False, use_bias=False):

    
    arch = None
    if model == 'flimunet':

        arch = utils.load_architecture(archpath)

        #input_shape = [H, W, C] or [C]
        encoder = utils.build_model(arch, input_shape=[3])
        #encoder = utils.load_weights_from_lids_model(encoder, parampath+ "/flair")


        encoder2 = utils.build_model(arch, input_shape=[3])
        #encoder2 = utils.load_weights_from_lids_model(encoder2,parampath+ "/t1gd")

        if parampath is not None:
            encoder = utils.load_weights_from_lids_model(encoder, parampath+ "/flair")
            encoder2 = utils.load_weights_from_lids_model(encoder2,parampath+ "/t1gd")
        
            if use_bias:
                encoder  = load_first_layer_bias(encoder, parampath + "/flair")
                encoder2 = load_first_layer_bias(encoder2, parampath + "/t1gd") 

                #exit()


        net = UNet(encoder1=encoder, encoder2=encoder2, out_channels=out_channels, train_encoder=train_encoder)

    elif model == 'resunet':
        net = ResUNet(num_classes=4, in_channels=2, depth=network_depth)
    
    elif model == 'standard_unet':
        net  = StandUNet(num_classes=4, in_channels=2, depth=network_depth) 
    else:
        raise NotImplementedError(f'there is no model  {model}')

    return net, arch


def getInsideModel1enc(model='resunet', out_channels=4, archpath=None, parampath=None, network_depth=64,
                   train_encoder=False, use_bias=False):

    
    arch = None
    if model == 'flimunet':

        arch = utils.load_architecture(archpath)

        #input_shape = [H, W, C] or [C]
        encoder = utils.build_model(arch, input_shape=[2])

        if parampath is not None:
            encoder = utils.load_weights_from_lids_model(encoder, parampath)
            if use_bias:
                encoder  = load_first_layer_bias(encoder, parampath)

        net = UNet1enc(encoder1=encoder, encoder2=None, out_channels=out_channels, train_encoder=train_encoder)

    else:
        raise NotImplementedError(f'there is no model  {model}')

    return net, arch

def run_experiment(datapath='/app/data', batchsize=1, archpath='/app/arch.json',
                   parampath='/app/param',
                   datatype='brats', modeltype='flimunet',
                   n_epochs=30, exp_name='test', lr=2.5e-4,train_encoder=False,
                   use_loss='CE', use_bias=False, oneEnc=False):

    if oneEnc:
        insidemodel,arch = getInsideModel1enc(modeltype, out_channels=4, archpath=archpath, parampath=parampath,
                                 train_encoder=train_encoder,use_bias=use_bias)
    else:
        insidemodel,arch = getInsideModel(modeltype, out_channels=4, archpath=archpath, parampath=parampath,
                                 train_encoder=train_encoder, use_bias=use_bias)

    #print(insidemodel)
    #exit()
    model = LitModel(insidemodel, optim='adam', lr=lr, use_loss=use_loss, batch_size=batchsize)

    transform = transforms.Compose([ToTensor()])
    trn_dl, val_dl, test_dl = getDataloaders(datatype, datapath, transform, batchsize, modeltype, num_workers=0, oneEnc=oneEnc)
    
    model_checkpoint = ModelCheckpoint(
        monitor='val_ALL_dice',
        dirpath='exp/',
        filename=exp_name + '{epoch:02d}-{val_loss:.2f}-{val_WT_dice:.2f}_dice',
        save_top_k=2,
        mode='max',
    )

    logger = TensorBoardLogger('exp/logs', name=exp_name)
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=[3],
        precision=16,
        accumulate_grad_batches=1,#accum_batch_size,
        max_epochs=n_epochs,
        logger=logger,
        #gradient_clip_val=0.1,
        log_every_n_steps=int(49/batchsize),
        callbacks=[LearningRateMonitor(logging_interval='step'), model_checkpoint],
    )
    # training
    trainer.fit(model, trn_dl, val_dl)

    trainer.save_checkpoint(f'exp/{exp_name}_final.ckpt')

    print(f"best model {model_checkpoint.best_model_path}")

    trainer.test(model, trn_dl, ckpt_path=model_checkpoint.best_model_path)
    trainer.test(model, val_dl, ckpt_path=model_checkpoint.best_model_path)
    trainer.test(model, test_dl, ckpt_path=model_checkpoint.best_model_path)

    #model = LitModel.load_from_checkpoint(model_checkpoint.best_model_path)

    checkpoint = th.load(model_checkpoint.best_model_path)
    print("best model is",model_checkpoint.best_model_path)
    model.load_state_dict(checkpoint['state_dict'])

    print(f"parampath is {parampath}")

    #if arch is not None:
        #utils.save_lids_model(insidemodel.encoder1, arch, 'output_dir', exp_name + "/flair")
        #utils.save_lids_model(insidemodel.encoder2, arch, 'output_dir', exp_name + "/t1gd")


    '''
    count=0
    for step, batch in enumerate(val_dl):
        xf, xt, gt = batch[0], batch[1], batch[2]
        model.eval()
        y_hat = model.forward(xf, xt)
        for i in range(y_hat.shape[0]):
            save_predition(y_hat[i], batch, './out', count)
            count+=1
    '''
    print("final")


if __name__ == '__main__':

    epochs   = 1

    for i in range(1):

        run_experiment(datapath='/app/data/glioblastoma/rigid/100',batchsize=1,
                    archpath='/app/data/archs/new_small/arch.json',
                    parampath='/app/data/ms_files/bkp_models/2enc/new_model_old_t1_adjusted/',
                    datatype='ours', modeltype='flimunet',lr=2.5e-4,
                    n_epochs=int(epochs), exp_name='FLIM_adj_FIX_bothLoss_e-4_b1_1gpu_poly',
                    train_encoder=False, use_loss="BOTH", use_bias=False)


        run_experiment(datapath='/app/data/glioblastoma/rigid/100',batchsize=1,
                    archpath='/app/data/archs/new_small/arch.json',
                    parampath='/app/data/ms_files/bkp_models/2enc_bias/biased_model_2encoders',
                    datatype='ours', modeltype='flimunet',lr=2.5e-4,
                    n_epochs=int(epochs), exp_name='biased_bias_adj_FIX_bothLoss_e-4_b1_1gpu_poly',
                    train_encoder=False, use_loss="BOTH", use_bias=True)

        # one encoder NO bias
        run_experiment(datapath='/app/data/glioblastoma/rigid/100',batchsize=1,
                    archpath='/app/data/archs/small_1encoder/arch.json',
                    parampath='/app/data/ms_files/bkp_models/1enc/1_encoder/model_dez/',
                    datatype='ours', modeltype='flimunet',lr=2.5e-4,
                    n_epochs=int(epochs), exp_name='1encoder_dez',
                    train_encoder=False, use_loss="BOTH", use_bias=False,oneEnc=True)

        # one encoder bias
        run_experiment(datapath='/app/data/glioblastoma/rigid/100',batchsize=1,
                    archpath='/app/data/archs/small_1encoder/arch.json',
                    parampath='/app/data/ms_files/bkp_models/1enc_bias/bias_1_encoder_new/bias_merge_selected/',
                    datatype='ours', modeltype='flimunet',lr=2.5e-4,
                    n_epochs=int(epochs), exp_name='1encoder_bias',
                    train_encoder=False, use_loss="BOTH", use_bias=True,oneEnc=True)
        


