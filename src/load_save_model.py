from flim.experiments import utils

import numpy as np

import torch

import sys
import os

arch = utils.load_architecture(sys.argv[1])
parampath = sys.argv[2]

encoder = utils.build_model(arch, input_shape=[3])
encoder = utils.load_weights_from_lids_model(encoder, parampath)

utils.save_lids_model(encoder, arch, 'output_dir', 'model0')

