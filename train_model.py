import warnings
warnings.filterwarnings('ignore')

import unet
import dlutil
from dlutil.data import ToFloat
from dlutil.data import get_index_batches
from dlutil.data import BatchIndexSampler
from dlutil.data import aggregate_classes
from dlutil.utils import y_vis_sample, from_onehot, _make_agg_matrix

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# these are required for defining the regression ImageFolder
from typing import Dict, Any
from torchvision import datasets
from torchsummary import summary

import nilearn as ni
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

import os
import copy
import pickle
import collections

import numpy as np
import pandas as pd

from PIL import Image
from PIL import ImageOps
from PIL import ImagePalette

import nbutils
import datetime

# ---- DEFINE GLOBAL VARIABLES AND DIRECTORIES ---- #
# ---- DATA LOADING ---- #

train = True
reload_parameters = True
reload_optimizer_state = True
warmstart = True

date = str(datetime.datetime.now().date())
model_save_dir = f'./models/model_{date}'
os.makedirs(model_save_dir, exist_ok=True)

base_data_dir = '/scratch/groups/jyeatman/samjohns-projects/data/atlas/ots-sulc'
data_prefix = 'ots'
xdir = os.path.join(base_data_dir, 'xs')
ydir = os.path.join(base_data_dir, 'ys')

trn_csv_fpath = f'{base_data_dir}/{data_prefix}_sulc_mapping_trn.csv'
val_csv_fpath = f'{base_data_dir}/{data_prefix}_sulc_mapping_val.csv'

trn_df = pd.read_csv(trn_csv_fpath)
val_df = pd.read_csv(val_csv_fpath)

trn_df['Train'] = True
val_df['Train'] = False
data_df = pd.concat([trn_df, val_df])

out_channels = 6
fg_agg_dict = {0: [0], 1: list(range(1, out_channels))}
n_agg_classes = len(fg_agg_dict)

null_agg_dict = {i: [i] for i in range(out_channels)}
ots_agg_dict = {0: [0], 1: [1], 2: [2, 3, 4, 5]}

trf = transforms.Compose([transforms.ToTensor(),
                          dlutil.data.ToFloat()])
target_trf = transforms.Compose([transforms.ToTensor(),
                                 dlutil.data.ToLong()])

ds_trn = dlutil.data.CustomUnetDataset(xdir=xdir, ydir=ydir,
                                       mapping_file=trn_csv_fpath,
                                       transform=trf,
                                       target_transform=target_trf,
                                       nclasses=out_channels,
                                       agg_dict=ots_agg_dict)

ds_val = dlutil.data.CustomUnetDataset(xdir=xdir, ydir=ydir,
                                       mapping_file=val_csv_fpath,
                                       transform=trf,
                                       target_transform=target_trf,
                                       nclasses=out_channels,
                                       agg_dict=ots_agg_dict)

# create dataloaders
bs = 32
nw = 3  # number of cpu's to use to parallelize dl
pm = True  # whether to pin memory in dl

dl_train = DataLoader(ds_trn, batch_size=bs, shuffle=True, pin_memory=pm)
dl_test = DataLoader(ds_val, batch_size=bs, shuffle=False, pin_memory=pm)

# ---- DEFINE MODEL ---- #

# get data shape
xtr, ytr = next(iter(dl_train))
xte, yte = next(iter(dl_test))
in_channels = xte.shape[1]
out_channels_agg = yte.shape[1]

# define model and move to device (GPU if available)
model = unet.UNet(feature_count=in_channels,
                  class_count=out_channels_agg,
                  apply_sigmoid=False)
device = dlutil.utils.get_device()
model.to(device)

# load model parameters
def get_last_fns(model_load_dir, return_optim=True):
    """
        Selects the latest model and load files
        from a given directory, returning them
        as a tuple of filenames.
    """
    import os
    fns = os.listdir(model_load_dir)
    opt = []
    if return_optim:
        opt = sorted([fn for fn in fns if 'optim' in fn])[-1]
    mod = sorted([fn for fn in fns if 'model' in fn])[-1]
    return mod, opt


def make_pretrained_state_dict(model_state, pretrained_state):
    """
        Given a current model state dictionary (model_state),
        and another state_dict, with *all* keys in model_state,
        returns a copy of mode_state with all parameters named
        in pretrained_state set to their values there.

        Allows, e.g., all but last layer of a network to be initialized
        from a previously trained model with different last layer architecture.
    """
    warmstart_params = copy.deepcopy(model_state)
    for k, v in pretrained_state.items():
        warmstart_params[k] = v
    return warmstart_params


if reload_parameters:

    cpu = (device.type == 'cpu')
    # in case model is saved from GPU but reloaded to CPU
    device_string = 'cuda' if not cpu else 'cpu'

    model_load_dir = './models/model_2022-10-26'
    model_state_fn, optim_state_fn = get_last_fns(model_load_dir)

    # reload model state dict
    with open(f'{model_load_dir}/{model_state_fn}', 'rb') as f:
        model_reload_dict = torch.load(f, map_location=torch.device(device_string))

    # reload optimizer state dict
    with open(f'{model_load_dir}/{optim_state_fn}', 'rb') as f:
        optim_reload_dict = torch.load(f, map_location=torch.device(device_string))


pretrained_params = model_reload_dict


def make_warmstart_state_dict(model,
                              pretrained_params,
                              discard_keywords=None):
    """
        Creates a state dict suitable for initializing a model whose
        architecture partially overlaps with that of another, already-trained
        model.

        Note: to save space, this function WILL modify the dictionary
            pretrained_params *IN PLACE* by popping some elements!

        Arguments:
            model : the PyTorch model to initialize
            pretrained_params : the (possibly incompatible) state dict of
                another model, which is to be partially pasted onto the model's
                own parameters.
            discard_keywords : keywords
        Returns:
            params : the joined parameters ready to be loaded into new model
    """
    if discard_keywords is None:
        discard_keywords = ['conv_last']
    discard_keys = []
    for keyword in discard_keywords:
        discard_keys += [k for k in list(pretrained_params.keys()) if
                         keyword in k]
    for k in discard_keys:
        pretrained_params.pop(k, None)
    params = make_pretrained_state_dict(model.state_dict(), pretrained_params)
    return params


if warmstart:
    params = make_warmstart_state_dict(model, pretrained_params)
else:
    params = pretrained_params

model.load_state_dict(params)

# freeze all but the last layer
dlutil.utils.set_unfreeze_(model, [model.conv_last])
# confirm correct freeze / unfreeze
assert dlutil.utils.check_requires_grad(model.conv_last) and not \
       dlutil.utils.check_requires_grad(model.conv_up3)

# data dictionaries formatted for input to training loop
dl_dict = {'trn': dl_train, 'val': dl_test}

# training proceeds along this plan, in sequential phases with different params.
# training plan 1: multi-phase, not used by default
training_plan = [
    dict(lr=0.00375, gamma=0.9, num_batches=int(1e4), bce_weight=0.67),
    dict(lr=0.00250, gamma=0.9, num_batches=int(1e4), bce_weight=0.33),
    dict(lr=0.00125, gamma=0.9, num_batches=int(1e4), bce_weight=0.00)
]
# training plan 2: single-phase, used by default
training_plan_short = [dict(lr=0.00375, gamma=0.9,
                            num_batches=int(1e5),
                            bce_weight=0.33)]

# ---- TRAIN MODEL ---- #

if train:

    tp = training_plan_short  # abbreviated for testing
    for e, t in enumerate(tp):

        # unpack hyperparameters
        lr = t['lr']
        gamma = t['gamma']
        bce_weight = t['bce_weight']
        num_batches = t['num_batches']

        # define training objects
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

        # initialize optimizer
        if reload_optimizer_state and (e == 0):
            optimizer.load_state_dict(optim_reload_dict)

        # other parameters
        step_size = len(dl_dict['trn'])
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma)

        # train loop
        result = unet.train.train_model(model,
                                        optimizer,
                                        scheduler,
                                        dl_dict,
                                        save_path=model_save_dir,
                                        report_every=5,
                                        checkpoint_every=100,
                                        logits=True,
                                        num_batches=num_batches,
                                        logger=None)
        model, best_dice, best_bce, losses = result


