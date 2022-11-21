import os
import argparse

import warnings
warnings.filterwarnings('ignore')

import unet
import dlutil
from dlutil.data import ToFloat
from dlutil.utils import get_last_fns
from dlutil.utils import make_warmstart_state_dict
from dlutil.utils import make_pretrained_state_dict

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import nilearn as ni
import matplotlib.pyplot as plt
import copy
import pickle
import collections

import numpy as np
import pandas as pd
from run_config import Config

# ---- DATA LOADING ---- #


def main(args):
    """
        1. Read configuration file.
        2. Load data from config-specified directories, and package into
            training and validation dataloaders.
        3. Train a model, saving incremental checkpoints.
    """
    # options
    config = Config(args.config_file)

    trn_df = pd.read_csv(config.trn_csv_fpath)
    val_df = pd.read_csv(config.val_csv_fpath)
    trn_df['Train'] = True
    val_df['Train'] = False
    data_df = pd.concat([trn_df, val_df])

    out_channels = config.out_channels
    agg_dict = config.agg_dict

    trf = transforms.Compose([transforms.ToTensor(),
                              dlutil.data.ToFloat()])
    target_trf = transforms.Compose([transforms.ToTensor(),
                                     dlutil.data.ToLong()])

    xdir = config.xdir
    ydir = config.ydir

    # create datasets
    ds_trn = dlutil.data.CustomUnetDataset(xdir=xdir, ydir=ydir,
                                           mapping_file=config.trn_csv_fpath,
                                           transform=trf,
                                           target_transform=target_trf,
                                           nclasses=out_channels,
                                           agg_dict=agg_dict)

    ds_val = dlutil.data.CustomUnetDataset(xdir=xdir, ydir=ydir,
                                           mapping_file=config.val_csv_fpath,
                                           transform=trf,
                                           target_transform=target_trf,
                                           nclasses=out_channels,
                                           agg_dict=agg_dict)

    # create dataloaders
    dl_train = DataLoader(ds_trn,
                          batch_size=config.batch_size,
                          shuffle=True,
                          pin_memory=config.pin_memory,
                          num_workers=config.num_workers_in_dataloaders)
    dl_test = DataLoader(ds_val,
                         batch_size=config.batch_size,
                         shuffle=False,
                         pin_memory=config.pin_memory,
                         num_workers=config.num_workers_in_dataloaders)

    # ---- DEFINE MODEL ---- #

    # get data shape
    xtr, ytr = next(iter(dl_train))
    xte, yte = next(iter(dl_test))
    in_channels = xte.shape[1]  # 0th dim is batch
    out_channels_agg = yte.shape[1]  # 0th dim is batch

    # define model and move to device (GPU if available)
    model = unet.UNet(feature_count=in_channels,
                      class_count=out_channels_agg,
                      apply_sigmoid=False)
    device = dlutil.utils.get_device()
    model.to(device)

    if config.reload_parameters:

        cpu = (device.type == 'cpu')
        # in case model is saved from GPU but reloaded to CPU
        device_string = 'cuda' if not cpu else 'cpu'
        model_state_fn, optim_state_fn = get_last_fns(config.model_load_dir)

        # reload model state dict
        with open(f'{config.model_load_dir}/{model_state_fn}', 'rb') as f:
            model_reload_dict = \
                torch.load(f, map_location=torch.device(device_string))

        # reload optimizer state dict
        with open(f'{config.model_load_dir}/{optim_state_fn}', 'rb') as f:
            optim_reload_dict = \
                torch.load(f, map_location=torch.device(device_string))

    if config.warmstart:
        params = make_warmstart_state_dict(model, model_reload_dict)
    else:
        params = model_reload_dict

    model.load_state_dict(params)

    # freeze all but the last layer
    dlutil.utils.set_unfreeze_(model, [model.conv_last])
    # confirm correct freeze / unfreeze
    assert dlutil.utils.check_requires_grad(model.conv_last) and not \
           dlutil.utils.check_requires_grad(model.conv_up3)

    # data dictionaries formatted for input to training loop
    dl_dict = {'trn': dl_train, 'val': dl_test}

    # ---- TRAIN MODEL ---- #

    if config.train:

        for e, t in enumerate(config.training_plan):

            # unpack hyperparameters
            lr = t['lr']
            gamma = t['gamma']
            bce_weight = t['bce_weight']
            num_batches = t['num_batches']

            # define training objects
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

            # initialize optimizer
            if config.reload_optimizer_state and (e == 0):
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
                                            save_path=config.model_save_dir,
                                            report_every=5,
                                            checkpoint_every=100,
                                            logits=True,
                                            num_batches=num_batches,
                                            logger=None)
            model, best_dice, best_bce, losses = result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default=None)
    parser.add_argument("--train", type=bool, default=True)
    main(parser.parse_args())
