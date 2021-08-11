from flim.experiments import utils

import numpy as np

import torch

from torch import optim

from torchvision import transforms

from torch_snippets import *

from model import UNet, UnetLoss, train_batch, validate_batch, get_device, IoU
from dataloader import SegmDataset, ToTensor


def run_experiment(datapath='/app/data', batchsize=1, archpath='/app/arch.json',
                   parampath='brain3d-param-small', n_epochs=30):

    device = get_device()

    arch = utils.load_architecture(archpath)

    #input_shape = [H, W, C] or [C]
    encoder = utils.build_model(arch, input_shape=[3])

    encoder = utils.load_weights_from_lids_model(encoder, parampath)

    num_classes = 2
    u_net = UNet(encoder=encoder, out_channels=num_classes)

    model = u_net.to(device)
    criterion = UnetLoss

    optimizer = optim.Adam(model.decoder.parameters(), lr=1e-3)

    transform = transforms.Compose([ToTensor()])

    #brain_path = 'brain3d_50'

    trn_ds = SegmDataset(datapath, transform=transform, train=True, gts=True)
    val_ds = SegmDataset(datapath, transform=transform, train=False, gts=True)


    trn_dl = DataLoader(trn_ds, batch_size=batchsize, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batchsize, shuffle=True)

    log = Report(n_epochs)
    for ex in range(n_epochs):
        all_preds = None
        all_true_labels = None
        
        model.decoder.train()
        N = len(trn_dl)
        for bx, data in enumerate(trn_dl):
            loss, acc = train_batch(model, data, optimizer, criterion, device=device)
            log.record(ex+(bx+1)/N, trn_loss=loss, trn_acc=acc, end='\r')

        model.decoder.eval()
        N = len(val_dl)
        for bx, data in enumerate(val_dl):
            loss, acc, preds, true_labels = validate_batch(model, data, criterion, device=device)
            log.record(ex+(bx+1)/N, val_loss=loss, val_acc=acc, end='\r')

            if all_preds is None:
                all_preds = preds.detach().cpu().numpy().flatten()
                all_true_labels = true_labels.detach().cpu().numpy().flatten()
            else:
                all_preds = np.concatenate((all_preds, preds.detach().cpu().numpy().flatten()))
                all_true_labels = np.concatenate((all_true_labels, true_labels.detach().cpu().numpy().flatten()))
            
        log.report_avgs(ex+1)

        print("IoU of validation set", IoU(all_true_labels, all_preds, ignore_label=2))




if __name__ == '__main__':

    run_experiment(datapath='/dados/matheus/git/u-net-with-flim2/brain3d_50',batchsize=1, archpath='/dados/matheus/git/u-net-with-flim2/archift3d.json',
                    parampath='/dados/matheus/git/u-net-with-flim2/brain3d-param',
                    n_epochs=1)