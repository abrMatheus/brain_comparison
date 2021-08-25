import re
from typing import Union, Sequence, Optional, Callable
from pathlib import Path

import torchio as tio
from torchio.data import SubjectsDataset, Subject
import torchio.transforms as tr


def get_test_transforms(mode: str) -> Callable:
    assert mode in ('basic', 'quantnorm', 'aug')

    percentiles = (0, 100) if mode == 'basic' else (0.5, 99.5)
    return tr.Compose([
        tr.RescaleIntensity(percentiles=percentiles, exclude='seg'),
        tr.RemapLabels({0:0, 1:1, 2:2, 4:3}, include='seg'),
        tr.Resample([2,2,2])
    ])


def get_train_transforms(mode: str) -> Callable:
    if mode == 'aug':
        trans = [
            tr.RandomFlip(('Left',)),
            tr.RandomElasticDeformation(),
            tr.RandomBiasField(exclude='seg', p=0.5),
            tr.RandomGhosting(exclude='seg', p=0.5),
        ]
    else:
        trans = []
        
    trans += get_test_transforms(mode).transforms

    return tr.Compose(trans)



class BratsDataset(SubjectsDataset):
    subdir = {
        'train': 'MICCAI_BraTS2020_TrainingData',
        'val': 'MICCAI_BraTS2020_ValidationData',
    }
    label_key = 'seg'

    def __init__(self,
                 directory: Union[str, Path],
                 mode: str,
                 transform: Optional[Callable] = None,
                 load_getitem: bool = True):

        if isinstance(directory, str):
            directory = Path(directory)

        self.directory = directory / self.subdir[mode]
        if not self.directory.exists():
            raise ValueError(f'Directory {self.directory} not found.')

        subjects = self._load_subjects(self.directory)

        super().__init__(subjects, transform=transform, load_getitem=load_getitem)

    def _subdir_to_subject(self, subdir: Path) -> Subject:
        images = {}
        for im_path in subdir.iterdir():
            key = re.findall(r'[a-z1-9]+(?=\.nii\.gz)', im_path.name)[0]

            if key == self.label_key:
                images[key] = tio.LabelMap(im_path)
            else:
                images[key] = tio.ScalarImage(im_path)

        return Subject(**images)

    def _load_subjects(self, directory: Path) -> Sequence[Subject]:
        subjects = []
        for subdir in directory.glob('BraTS20_*'):
            if not subdir.is_dir():
                continue
            subjects.append(self._subdir_to_subject(subdir))

        return subjects
