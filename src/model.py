from flim.experiments import utils

import numpy as np

import torch

from torch import nn

from torch import optim

from torchvision import transforms

from torch_snippets import *

from sklearn.metrics import jaccard_score

from PIL import Image

import functools

from collections import OrderedDict

from flashtorch.activmax import GradientAscent

from dice_loss import SoftDiceLoss, SoftDiceLossSquared

def get_device():
    gpu = torch.cuda.is_available()

    if not gpu:
        device = torch.device('cpu')
    else:
        device = torch.device(1)

    return device




# U-Net
def rgetattr(obj, attr, *args):
    
    def _getattr (obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split('.'))

class IntermediateLayerGetter:
    def __init__(self, model, return_layers):
        self._model = model
        self._return_layers = return_layers

    def __call__(self, x):
        outputs = OrderedDict()
        handles = []

        for name, out_name in self._return_layers.items():
            layer = rgetattr(self._model, name)

            def hook(module, input, output, out_name=out_name):
                outputs[out_name] = output

            handle = layer.register_forward_hook(hook)

            handles.append(handle)
        
        self._model(x)

        for handle in handles:
            handle.remove()

        return outputs

def conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True)
    )

def up_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2),
        nn.ReLU(inplace=True)
    )
    
def _layers_before_downscale(model):
    last_layer_name = None
    last_out_channel = None
    layer_names = []
    last_out_channels = []
    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Sequential)):
            continue
        if isinstance(layer, nn.Conv3d):
            last_out_channel = layer.out_channels
        if isinstance(layer, (nn.AvgPool3d, nn.MaxPool3d)):
            layer_names.append(last_layer_name)
            last_out_channels.append(last_out_channel)
        last_layer_name = name

    layer_names.append(last_layer_name)
    last_out_channels.append(last_out_channel)

    return layer_names, last_out_channels

class UNet(nn.Module):
    def __init__(self, encoder1, encoder2, out_channels=2,
                 train_encoder=False):
        super().__init__()

        self.train_encoder=train_encoder

        
        #111111
        self.encoder1 = encoder1

        encoder_block_names1, block_out_channels1 = _layers_before_downscale(encoder1)

        layer_names = {layer_name: layer_name for layer_name in encoder_block_names1[:-1]}
        layer_names[encoder_block_names1[-1]] = "bottleneck"

        self._encoder_blocks1 = IntermediateLayerGetter(self.encoder1, layer_names)

        
            
        #2222222
        self.encoder2 = encoder2

        encoder_block_names2, block_out_channels2 = _layers_before_downscale(encoder2)

        layer_names = {layer_name: layer_name for layer_name in encoder_block_names2[:-1]}
        layer_names[encoder_block_names2[-1]] = "bottleneck"

        self._encoder_blocks2 = IntermediateLayerGetter(self.encoder2, layer_names)
        
        
        if self.train_encoder == False:
            self.encoder_eval()
        else:
            self.freeze_mnorm()
        
        
        self.decoder = nn.Module()
        
        print(f"block output channels {block_out_channels1}")
        
        block_out_channels1 = [i*2 for i in block_out_channels1]
        
        print(f"block output channels {block_out_channels1}")
        
        bottleneck_out_channels = block_out_channels1[-1] * 2 
        self.decoder.add_module("conv_bottleneck", conv(block_out_channels1[-1], bottleneck_out_channels))

        last_conv_out_channels = bottleneck_out_channels
        for i, _out_channels in enumerate(reversed(block_out_channels1[:-1])):

            self.decoder.add_module(f"up_conv{i}", up_conv(last_conv_out_channels, last_conv_out_channels//2))
            self.decoder.add_module(f"conv{i}", conv(last_conv_out_channels//2 + _out_channels, _out_channels))

            last_conv_out_channels = _out_channels
        
        self.decoder.add_module("output_layer", nn.Conv3d(last_conv_out_channels, out_channels, kernel_size=1))

    def freeze_mnorm(self):
        for param in self.encoder1.named_parameters():
            #print("param", param)
            if 'm-norm' in param[0]:
                param[1].requires_grad = False
        for param in self.encoder2.named_parameters():
            if 'm-norm' in param[0]:
                param[1].requires_grad = False

    def encoder_eval(self):
        for param in self.encoder1.parameters():
            #print("param", param)
            param.requires_grad = False
        for param in self.encoder2.parameters():
            param.requires_grad = False

    def forward(self, x1, x2):

        if self.train_encoder==False:
            self.encoder1.eval()
            self.encoder2.eval()
            #print(f"testing {self.training} {self.encoder1.training} {self.encoder2.training}")
        
            with torch.no_grad():
                encoder_outputs1 = self._encoder_blocks1(x1)
                encoder_outputs2 = self._encoder_blocks2(x2)

        else:
            encoder_outputs1 = self._encoder_blocks1(x1)
            encoder_outputs2 = self._encoder_blocks2(x2)

        block_names = reversed(encoder_outputs1.keys())

        block_name = next(block_names)
        bottleneck1 = encoder_outputs1[block_name]
        bottleneck2 = encoder_outputs2[block_name]

        #print(f'bneck1.shape {bottleneck1.shape} bneck2.shape {bottleneck2.shape}')
        
        x = torch.cat([bottleneck1, bottleneck2], dim=1)
        #print(f'x.shape {x.shape}')
            
            
        for name, layer in self.decoder.named_children():
            if "up_conv" in name:
                block_name = next(block_names)
                block_output1 = encoder_outputs1[block_name]
                block_output2 = encoder_outputs2[block_name]
                x = layer(x)
                block_output = torch.cat([block_output1, block_output2], dim=1)
                if (not x.shape == block_output):
                    #caso tenha problemas com a diferença de tamanhos
                    diff=np.array(x.shape) - np.array(block_output.shape)
                    p1d = (0, diff[4], 0, diff[3], 0, diff[2], 0, 0, 0, 0)
                    block_output = nn.functional.pad(block_output, p1d, "constant", 0)
                    
                x = torch.cat([x, block_output], dim=1)
                #print(f'name {name} layer {layer}')
                #print(f'x.shape {x.shape}, b1.shape {block_output1.shape} b2.shape {block_output2.shape}')
            else:
                x = layer(x)

        return x

# metrics
def IoU(gt, pred, ignore_label=-1, average='binary'):
    mask = gt != ignore_label
    iou = jaccard_score(gt[mask].flatten(), pred[mask].flatten(), average=average)
    return iou


weights = [0.1, 1., 1., 1.]
# class_weights = torch.FloatTensor(weights).cuda(1)
# ce = nn.CrossEntropyLoss(weight= class_weights).cuda(1)

class_weights = torch.FloatTensor(weights)
ce = nn.CrossEntropyLoss(weight = class_weights)

##ce = nn.CrossEntropyLoss().cuda(2)

##ce = nn.CrossEntropyLoss(ignore_index=2).to(device)

dice_loss = SoftDiceLoss(apply_nonlin=torch.sigmoid, **{'batch_dice': False, 'do_bg': True, 'smooth': 0})

def UnetLoss(preds, targets, use_loss='CE'):
    new_target = targets.clone()
    
    #if(not new_target.shape == preds.shape):
    #    diff=np.array(preds.shape) - np.array(new_target.shape)
    #    p1d = (0, diff[4], 0, diff[3], 0, diff[2], 0, 0, 0, 0)
    #    new_target = nn.functional.pad(new_target, p1d, "constant", 0)
    
    
    new_target = new_target.long()
    #print(f"preds type {type(preds)}, new_target {type(new_target)}")
    #print(f"shape {preds.shape} {new_target.shape}")
    #print(f"dtype {preds.dtype} {new_target.dtype}")
    
    gpu_num = preds.get_device()

    ce = nn.CrossEntropyLoss(weight = class_weights).cuda(gpu_num)
    
    ce_loss = ce(preds, new_target)
    ds_loss = dice_loss(preds, new_target)

    
    if use_loss=='CE':
        loss = ce_loss
    elif use_loss=='DS':
        loss = ds_loss
    elif use_loss == 'BOTH':
        loss = ce_loss + ds_loss
    else:
        raise NotImplementedError(f'UnetLoss - there is no loss  {use_loss}')

    pred_labels = torch.max(preds, 1)[1]
    mask = new_target != 10 
    
    
    acc = (pred_labels[mask] == new_target[mask]).float().mean()
    return loss, acc

def _maybe_resize(x, shape):
        if x.shape[-3:] != shape[-3:]:
            x = F.interpolate(x, size=shape[-3:],
                            mode="trilinear", align_corners=True)
        return x

from metrics.metrics import multiclass_dice, multiclass_sensitivity

def train_batch(model, data, optimizer, criterion, device='cpu', use_loss='CE'):
    flair, t1gd, y = data[0].to(device), data[1].to(device) , data[2].to(device)
    y_hat = model(flair, t1gd)

    y_hat = _maybe_resize(y_hat, y.shape)

    optimizer.zero_grad()
    loss, acc = criterion(y_hat, y, use_loss)
    loss.backward()
    optimizer.step()

    torch.cuda.empty_cache()

    logits = th.softmax(y_hat, dim=1)
    dice   = multiclass_dice(logits, y)

    return loss.item(), acc.item(), dice


@torch.no_grad()
def validate_batch(model, data, criterion, device='cpu', use_loss='CE'):
    flair, t1gd, y = data[0].to(device), data[1].to(device) , data[2].to(device)
    y_hat = model(flair, t1gd)

    y_hat = _maybe_resize(y_hat, y.shape)

    loss, acc = criterion(y_hat, y, use_loss)

    torch.cuda.empty_cache()

    logits = th.softmax(y_hat, dim=1)
    dice   = multiclass_dice(logits, y)

    return loss.item(), acc.item(), dice


class UNet1enc(nn.Module):
    def __init__(self, encoder1, encoder2, out_channels=2,
                 train_encoder=False):
        super().__init__()

        self.train_encoder=train_encoder

        
        #111111
        self.encoder1 = encoder1

        encoder_block_names1, block_out_channels1 = _layers_before_downscale(encoder1)

        layer_names = {layer_name: layer_name for layer_name in encoder_block_names1[:-1]}
        layer_names[encoder_block_names1[-1]] = "bottleneck"

        self._encoder_blocks1 = IntermediateLayerGetter(self.encoder1, layer_names)

        
        '''    
        #2222222
        self.encoder2 = encoder2

        encoder_block_names2, block_out_channels2 = _layers_before_downscale(encoder2)

        layer_names = {layer_name: layer_name for layer_name in encoder_block_names2[:-1]}
        layer_names[encoder_block_names2[-1]] = "bottleneck"

        self._encoder_blocks2 = IntermediateLayerGetter(self.encoder2, layer_names)
        '''
        
        if self.train_encoder == False:
            self.encoder_eval()
        else:
            self.freeze_mnorm()
        
        
        self.decoder = nn.Module()
        
        print(f"block output channels {block_out_channels1}")
        
        block_out_channels1 = [i*1 for i in block_out_channels1]
        
        print(f"block output channels {block_out_channels1}")
        
        bottleneck_out_channels = block_out_channels1[-1] * 4 
        self.decoder.add_module("conv_bottleneck", conv(block_out_channels1[-1], bottleneck_out_channels))

        last_conv_out_channels = bottleneck_out_channels
        for i, _out_channels in enumerate(reversed(block_out_channels1[:-1])):

            self.decoder.add_module(f"up_conv{i}", up_conv(last_conv_out_channels, last_conv_out_channels//2))
            self.decoder.add_module(f"conv{i}", conv(last_conv_out_channels//2 + _out_channels, _out_channels))

            last_conv_out_channels = _out_channels
        
        self.decoder.add_module("output_layer", nn.Conv3d(last_conv_out_channels, out_channels, kernel_size=1))

    def freeze_mnorm(self):
        for param in self.encoder1.named_parameters():
            #print("param", param)
            if 'm-norm' in param[0]:
                param[1].requires_grad = False
        #for param in self.encoder2.named_parameters():
        #    if 'm-norm' in param[0]:
        #        param[1].requires_grad = False

    def encoder_eval(self):
        for param in self.encoder1.parameters():
            #print("param", param)
            param.requires_grad = False
        #for param in self.encoder2.parameters():
        #    param.requires_grad = False

    def forward(self, x1, x2):

        x = torch.cat([x1, x2], dim=1) 

        if self.train_encoder==False:
            self.encoder1.eval()
            #self.encoder2.eval()

            #print(f"testing {self.training} {self.encoder1.training} {self.encoder2.training}")
        
            with torch.no_grad():
                encoder_outputs1 = self._encoder_blocks1(x)
                #encoder_outputs2 = self._encoder_blocks2(x2)

        else:
            encoder_outputs1 = self._encoder_blocks1(x)
            #encoder_outputs2 = self._encoder_blocks2(x2)

        block_names = reversed(encoder_outputs1.keys())

        block_name = next(block_names)
        bottleneck1 = encoder_outputs1[block_name]
        #bottleneck2 = encoder_outputs2[block_name]

        #print(f'bneck1.shape {bottleneck1.shape} bneck2.shape {bottleneck2.shape}')
        
        #x = torch.cat([bottleneck1, bottleneck2], dim=1)
        x = bottleneck1 
        #print(f'x.shape {x.shape}')
            
            
        for name, layer in self.decoder.named_children():
            if "up_conv" in name:
                block_name = next(block_names)
                block_output1 = encoder_outputs1[block_name]
                #block_output2 = encoder_outputs2[block_name]
                x = layer(x)
                #block_output = torch.cat([block_output1, block_output2], dim=1)
                block_output = block_output1
                if (not x.shape == block_output):
                    #caso tenha problemas com a diferença de tamanhos
                    diff=np.array(x.shape) - np.array(block_output.shape)
                    p1d = (0, diff[4], 0, diff[3], 0, diff[2], 0, 0, 0, 0)
                    block_output = nn.functional.pad(block_output, p1d, "constant", 0)
                    
                x = torch.cat([x, block_output], dim=1)
                #print(f'name {name} layer {layer}')
                #print(f'x.shape {x.shape}, b1.shape {block_output1.shape} b2.shape {block_output2.shape}')
            else:
                x = layer(x)

        return x