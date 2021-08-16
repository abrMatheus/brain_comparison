# Dataset
import os

from os import path

import torch

from torch.utils.data import Dataset

from flim.experiments import utils

import numpy as np

import torchio as tio


class SegmDataset(Dataset):
    def __init__(self, root_dir, transform=None, train=True, gts=False):
        assert isinstance(root_dir, str) and len(root_dir) > 0,\
            "Invalid root_dir"

        if train:
            self._root_dir = os.path.join(root_dir, 'train')
        else:
            self._root_dir = os.path.join(root_dir, 'val')

        self._is_train = train
        self._transform = transform

        self._gts = gts or not train

        self._image_names = None
        self._markers_names = None
        self._gt_names = None

        self._load_dataset_info()

    def __getitem__(self, index):
        flair_path = os.path.join(self._root_dir, "flair", f"{self._image_names[index]}.nii.gz")
        t1gd_path = os.path.join(self._root_dir, "t1gd", f"{self._image_names[index]}.nii.gz")

        if self._gts:
            label_path = os.path.join(self._root_dir, "labelu", f"{self._gt_names[index]}.nii.gz")
        else:
            label_path = os.path.join(self._root_dir, "markers", f"{self._markers_names[index]}.txt")
    
        flair_img = utils.load_image(flair_path)
        flair_img = flair_img.transpose((3, 0, 1, 2))
        t1gd_img = utils.load_image(t1gd_path)
        t1gd_img = t1gd_img.transpose((3, 0, 1, 2))
        
        label_image = self._load_label_image(label_path)

        if self._gts:
            label_image[label_image == 255] = 1

        if not self._gts:
            label_image[label_image == 0] = 3
            label_image = label_image - 1

        if(self._transform):
            flair_img = self._transform(flair_img)
            t1gd_img = self._transform(t1gd_img)
            
        sample = (flair_img, t1gd_img, label_image.astype(np.int64))
        
        return sample 

    def _load_dataset_info(self):
        if path.exists(self._root_dir):
            flair_path = path.join(self._root_dir, "flair")
            t1gd_path = path.join(self._root_dir, "t1gd")
            if self._gts:
                gts_path = path.join(self._root_dir, "labelu")
            else:
                markers_path = path.join(self._root_dir, "markers")

            if path.exists(flair_path) and path.exists(t1gd_path):
                self._image_names = [int(name.split('.')[0]) for name in os.listdir(flair_path)]
                self._image_names.sort()

                if self._gts:
                    self._gt_names = [int(name.split('.')[0]) for name in os.listdir(gts_path)]
                    self._gt_names.sort()
                else:
                    self._markers_names = [int(name.split('.')[0]) for name in os.listdir(markers_path)]
                    self._markers_names.sort()

            elif not path.exists(flair_path):
                raise ValueError(f"{flair_path} does not exists")
            else:
                raise ValueError(f"{t1gd_path} does not exists")
        else:
            raise ValueError(f"{self._root_dir} does not exists, dumb ass")

    def __len__(self):
        return len(self._image_names)

    def _load_label_image(self, label_path):
        if label_path.endswith('.txt'):
            with open(label_path, 'r') as f:
                lines = f.readlines()
            label_infos = [int(info) for info in lines[0].split(" ")]

            #images dimensions are flipped
            image_shape = (label_infos[2], label_infos[1])
            label_image = np.zeros(image_shape, dtype=np.int)

            for line in lines[1:]:
                split_line = line.split(" ")
                y, x, label = int(split_line[0]), int(split_line[1]), int(split_line[3])
                label_image[x][y] = label
            
            assert (label_image != 0).sum() == label_infos[0], "There are zero markers. Be careful!!"
        else:
            
            label_image = np.array(tio.LabelMap(label_path))
            
        return label_image
        
class ToTensor(object):
    def __call__(self, sample):
        image = np.array(sample)
        #image = image.transpose((3, 0, 1, 2))
        
        return torch.from_numpy(image.copy()).float()