# -*- coding: utf-8 -*-
################################################################################
# visual_autolabel/util.py
# Utilities fo the visual_autolabel library.

"""
The `visual_autolabel.util` package contains utilities for use in and with the
`visual_autolabel` library.
"""

#===============================================================================
# Constants / Globals

# Global Config Items.
# from .config import (default_partition, sids)  # commented out, as unnecessary.
import torch
import torch.nn.functional as F
import numpy as np
from nilearn import surface
import pipeline_utilities as pu
from pipeline_utilities import DESTRIEUX_FILENAME
import os
import trimesh  # note: trimesh handles triangulations of freesurfer meshes



#===============================================================================
# Utility Functions

#-------------------------------------------------------------------------------
# Subject Partitions
# Code for dealing with partitions of training and validation subjects.

def get_device():
    """ Returns gpu if available, cpu otherwise."""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _tensor_to_number(t):
    """Returns the raw numerical data stored in a torch tensor."""
    return t.cpu().numpy()


def is_partition(obj):
    """Returns true if the given object is a subject partition, otherwise False.

    `is_partition(x)` returns `True` if `x` is a mapping with the keys `'trn'`
    and `'val'` or is a tuple with 2 elements. Otherwise, returns `False`.

    Parameters
    ----------
    obj : object
        The object whose quality as a subject partition is to be determined.

    Returns
    -------
    boolean
        `True` if `obj` represents a subject partition and `False` otherwise.
    """
    from collections.abc import Mapping
    return ((isinstance(obj, tuple) and len(obj) == 2) or
            (isinstance(obj, Mapping) and 'trn' in obj and 'val' in obj))


def trndata(obj):
    """Returns the training data of an object representing a subject partition.

    `trndata((trn_data, val_data))` returns `trn_data` (i.e., if given a tuple
    of length 2, `trndata` will return the first element).

    `trndata({'trn': trn_data, 'val': val_data})` also returns `trn_data`.

    See also: `valdata`

    Parameters
    ----------
    obj : mapping or tuple
        Either a dict-like object with the keys `'trn'` and `'val'` or a tuple
        with two elements `(trn, val)`.

    Returns
    -------
    object
        Either the first element of `obj` when `obj` is a tuple or `obj['trn']`
        when `obj` is a mapping.
    """
    if isinstance(obj, tuple):
        return obj[0]
    else:
        return obj['trn']


def valdata(obj):
    """Returns the validation data from a subject partition.

    `valdata((trn_data, val_data))` returns `val_data` (i.e., if given a tuple
    of length 2, `valdata` will return the second element).

    `valdata({'trn': trn_data, 'val': val_data})` also returns `val_data`.

    Parameters
    ----------
    obj : mapping or tuple
        Either a dict-like object with the keys `'trn'` and `'val'` or a tuple
        with two elements `(trn, val)`.

    Returns
    -------
    object
        Either the second element of `obj` when `obj` is a tuple or `obj['val']`
        when `obj` is a mapping.
    """
    if isinstance(obj, tuple):
        return obj[1]
    else:
        return obj['val']


def partition_id(obj):
    """Returns a string that uniquely represents a subject partition.

    Parameters
    ----------
    obj : tuple or mapping of a subject partition
        A mapping that contains the keys `'trn'` and `'val'` or a tuple with two
        elements, `(trn, val)`. Both `trn` and `val` must be either iterables of
        subject-ids, datasets with the attribute `sids`, or dataloaders whose
        datasets have th attribute `sids`.

    Returns
    -------
    str
        A hexadecimal string that uniquely represents the partition implied by
        the `obj` parameter.
    """
    from torch.utils.data import (DataLoader, Dataset)
    trndat = trndata(obj)
    valdat = valdata(obj)
    if isinstance(trndat, DataLoader): trndat = trndat.dataset
    if isinstance(valdat, DataLoader): valdat = valdat.dataset
    if isinstance(trndat, Dataset):    trndat = trndat.sids
    if isinstance(valdat, Dataset):    valdat = valdat.sids
    trn = [(sid,'1') for sid in obj[0]]
    val = [(sid,'0') for sid in obj[1]]
    sids = sorted(trn + val, key=lambda x:x[0])
    pid = int(''.join([x[1] for x in sids]), 2)
    return hex(pid)

# note: modified with dummy sids: not necesary here
# sids = list(range(10))  # dummy sids
# def partition(sids, how=default_partition):
#     """Partitions a list of subject-IDs into a training and validation set.
#  
#     `partition(sids, (frac_trn, frac_val))` returns `(trn_sids, val_sids)` where
#     the fraction `frac_trn` of the `sids` have been randomly placed in the
#     training seet and `frac_val` of the subjects have been placed in the 
#     validation set, randomly. The sum `frac_trn + frac_val` must be between 0
#     and 1.
# 
#     `partition(sids, (num_trn, num_val))` where `num_trn` and `num_val` are both
#     positive integers whose sum is less than or equal to `len(sids)` places
#     exactly the number of subject-IDs, randomly, in each category.
# 
#     partition(sids, idstring)` where `idstring` is a hexadecimal string returned
#     by `partition_id()` reproduces the original partition used to create the
#     string.
# 
#     Parameters
#     ----------
#     sids : list-like
#         A list, tuple, array, or iterable of subject identifiers. The
#         identifiers may be numers or strings, but they must be sortable.
#     how : tuple or str
#         Either a tuple `(trn, val)` containing either the fraction of training
#         and validation set members (`trn + val == 1`) or the (integer) 
#         count of training and validation set members (`trn + val == len(sids)`),
#         or a hexadecimal string created by `partition_id`
# 
#     Returns
#     -------
#     tuple of arrays
#         A tuple `(trn_sids, val_sids)` whose members are numpy arrays of the
#         subject-IDs in the training and validation sets, respectively.
#     """
#     import numpy as np
#     sids = np.asarray(sids)
#     n = len(sids)
#     if isinstance(how, tuple):
#         ntrn = trndata(how)
#         nval = valdata(how)
#         if isinstance(ntrn, float) and isinstance(nval, float):
#             if ntrn < 0 or nval < 0: raise ValueError("trn and val must be > 0")
#             nval = round(nval * n)
#             ntrn = round(ntrn * n)
#             tot = nval + ntrn
#             if tot != n: raise ValueError("partition requires trn + val == 1")
#         elif isinstance(ntrn, int) and isinstance(nval, int):
#             if ntrn < 0 or nval < 0: raise ValueError("trn and val must be > 0")
#             tot = ntrn + nval
#             if tot != n: 
#                 raise ValueError("partition requires trn + val == len(sids)")
#         elif isinstance(ntrn, np.ndarray) and isinstance(nval, np.ndarray):
#             a1 = np.unique(sids)
#             a2 = np.unique(np.concatenate([ntrn, nval]))
#             if np.array_equal(a1, a2) and len(a1) == len(sids):
#                 return (ntrn, nval)
#             else:
#                 raise ValueError("partitions must include all sids")
#         else: raise ValueError("trn and val must both be integers or floats")
#         val_sids = np.random.choice(sids, nval)
#         trn_sids = np.setdiff1d(sids, val_sids)
#     elif isinstance(how, str):
#         sids = np.sort(sids)
#         trn_ii = np.array([1 if s == '1' else 0 for s in '{0:b}'.format(how)],
#                           dtype=np.bool)
#         trn_sids = sids[trn_ii]
#         val_sids = sids[~trn_ii]
#     return (trn_sids, val_sids)

#-------------------------------------------------------------------------------
# Filters and PyTorch Modules
# Code for dealing with PyTorch filters and models.

def kernel_default_padding(kernel_size):
    """Returns an appropriate default padding for a kernel size.

    The returned size is `kernel_size // 2`, which will result in an output
    image the same size as the input image.

    Parameters
    ----------
    kernel_size : int or tuple of ints
        Either an integer kernel size or a tuple of `(rows, cols)`.

    Returns
    -------
    int
        If `kernel_size` is an integer, returns `kernel_size // 2`.
    tuple of ints
        If `kernel_size` is a 2-tuple of integers, returns
        `(kernel_size[0] // 2, kernel_size[1] // 2)`.
    """
    try:
        return (kernel_size[0] // 2, kernel_size[1] // 2)
    except TypeError:
        return kernel_size // 2


def convrelu(in_channels, out_channels,
             kernel=3, padding=None, stride=1, bias=True, inplace=True):
    """Shortcut for creating a PyTorch 2D convolution followed by a ReLU.

    Parameters
    ----------
    in_channels : int
        The number of input channels in the convolution.
    out_channels : int
        The number of output channels in the convolution.
    kernel : int, optional
        The kernel size for the convolution (default: 3).
    padding : int or None, optional
        The padding size for the convolution; if `None` (the default), then
        chooses a padding size that attempts to maintain the image-size.
    stride : int, optional
        The stride to use in the convolution (default: 1).
    bias : boolean, optional
        Whether the convolution has a learnable bias (default: True).
    inplace : boolean, optional
        Whether to perform the ReLU operation in-place (default: True).

    Returns
    -------
    torch.nn.Sequential
        The model of a 2D-convolution followed by a ReLU operation.
    """
    import torch
    if padding is None:
        padding = kernel_default_padding(kernel)
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_channels, out_channels, kernel,
                        padding=padding, bias=bias),
        torch.nn.ReLU(inplace=inplace))

#-------------------------------------------------------------------------------
# Loss Functions

def is_logits(data):
    """Attempts to guess whether the given PyTorch tensor contains logits.

    If the argument `data` contains only values that are no less than 0 and no
    greater than 1, then `False` is returned; otherwise, `True` is returned.
    """
    if   (data > 1).any(): return True
    elif (data < 0).any(): return True
    else:                  return False


def dice_loss(pred,
              gold,
              logits=None,
              smoothing=1,
              graph=False,
              class_weights=None,
              metrics=None
              ):
    """Returns the loss based on the dice coefficient.
    
    `dice_loss(pred, gold)` returns the dice-coefficient loss between the
    tensors `pred` and `gold` which must be the same shape and which should
    represent probabilities. The first two dimensions of both `pred` and `gold`
    must represent the batch-size and the classes.

    Parameters
    ----------
    pred : tensor
        The predicted probabilities of each class.
    gold : tensor
        The gold-standard labels for each class.
    logits : boolean, optional
        Whether the values in `pred` are logits--i.e., unnormalized scores that
        have not been run through a sigmoid calculation already. If this is
        `True`, then the BCE starts by calculating the sigmoid of the `pred`
        argument. If `None`, then attempts to deduce whether the input is or is
        not logits. The default is `None`.
    smoothing : number, optional
        The smoothing coefficient `s`. The default is `1`.
    class_weights : sequence of length nclasses. If provided, this reweights
        the importance of each class (according to its respective weight)
        in calculating the DICE loss.
    metrics : dict or None, optional
        An optional dictionary into which the key `'dice'` should be inserted
        with the dice-loss as the value.

    Returns
    -------
    float
        The dice-coefficient loss of the prediction.
    """
    import torch

    # parse class_weights arg (e.g., ensure normalization, convert to torch)
    if class_weights is not None:
        assert len(class_weights) == pred.shape[1]
        class_weights = torch.Tensor(class_weights)
    else:
        class_weights = torch.ones(pred.shape[1])
    class_weights = class_weights / torch.sum(class_weights)
    class_weights = class_weights.to(get_device())

    pred = pred.contiguous()
    gold = gold.contiguous()
    if logits is None:  # infer
        logits = is_logits(pred)
    if logits:  # user specified True
        pred = torch.sigmoid(pred)
    intersection = (pred * gold)
    pred = pred**2
    gold = gold**2
    while len(intersection.shape) > 2:
        intersection = intersection.sum(dim=-1)
        pred = pred.sum(dim=-1)
        gold = gold.sum(dim=-1)
    if smoothing is None:
        smoothing = 0
    # Calculate the DICE according to the mathematical definition.
    loss = (1 - ((2 * intersection + smoothing) / (pred + gold + smoothing)))
    # Average the loss across classes then take the mean across batch elements.
    loss = torch.matmul(loss, class_weights).mean()
    if metrics is not None:
        if 'dice' not in metrics: metrics['dice'] = 0.0
        metrics['dice'] += loss.data.cpu().numpy() * gold.size(0)
    return loss


def bce_loss(
    pred,
    gold,
    logits=None,
    reweight=False,
    class_weights=None,
    metrics=None
):
    """Returns the loss based on the binary cross entropy.
    
    `bce_loss(pred, gold)` returns the binary cross entropy loss between the
    tensors `pred` and `gold` which must be the same shape and which should
    represent probabilities. The first two dimensions of both `pred` and `gold`
    must represent the batch-size and the classes.

    Parameters
    ----------
    pred : tensor
        The predicted probabilities of each class.
    gold : tensor
        The gold-standard labels for each class.
    logits : boolean, optional
        Whether the values in `pred` are logits--i.e., unnormalized scores that
        have not been run through a sigmoid calculation already. If this is
        `True`, then the BCE starts by calculating the sigmoid of the `pred`
        argument. If `None`, then attempts to deduce whether the input is or is
        not logits. The default is `None`.
    reweight : boolean, optional
        Whether to reweight the classes by calculating the BCE for each class
        then calculating the mean across classes. If `False`, then the raw BCE
        across all pixels, classes, and batches is returned (the default).
    metrics : dict or None, optional
        An optional dictionary into which the key `'bce'` should be inserted
        with the bce-loss as the value.

    Returns
    -------
    float
        The binary cross entropy loss of the prediction.
    """
    import torch
    if logits is None:
        logits = is_logits(pred)
    if class_weights is not None:
        assert len(class_weights) == pred.shape[1]
        class_weights = torch.Tensor(class_weights)
        class_weights = class_weights / torch.sum(class_weights)
        class_weights = class_weights.to(get_device())
    if logits:
        f = torch.nn.functional.binary_cross_entropy_with_logits
    else:
        f = torch.nn.functional.binary_cross_entropy
    # manually weighted classes
    if class_weights is not None:
        r = 0
        for k in range(pred.shape[1]):  # class dim = dim 1
            p = pred[:, [k]]
            t = gold[:, [k]]
            r += f(p, t) * class_weights[k]
    # automatically weighted classes
    elif reweight:
        n = pred.shape[-1] * pred.shape[-2]
        # multiply normalization factor by batch size if data is batched
        if len(pred.shape) > 3:
            n *= pred.shape[0]
        r = 0
        for k in range(pred.shape[1]):  # class dim = dim 1
            p = pred[:, [k]]
            t = gold[:, [k]]
            r += f(p, t) * (n - torch.sum(t)) / n
    # no reweighting (easy)
    else:
        r = f(pred, gold)
    if metrics is not None:
        if 'bce' not in metrics:
            metrics['bce'] = 0.0
        metrics['bce'] += r.data.cpu().numpy() * gold.size(0)
    return r


def loss(
    pred,
    gold,
    logits=True,
    bce_weight=0.5,
    smoothing=1,
    reweight=True,
    class_weights=None,
    metrics=None
):
    """Returns the weighted sum of dice-coefficient and BCE-based losses.

    `loss(pred, gold)` calculates the loss value between the given prediction
    and gold-standard labels, both of which must be the same shape and whose
    elements should represent probability values.

    Parameters
    ----------
    pred : tensor
        The predicted probabilities of each class.
    gold : tensor
        The gold-standard labels for each class.
    logits : boolean, optional
        Whether the values in `pred` are logits--i.e., unnormalized scores that
        have not been run through a sigmoid calculation already. If this is
        `True`, then the BCE starts by calculating the sigmoid of the `pred`
        argument. If `None`, then attempts to deduce whether the input is or is
        not logits. The default is `None`.
    bce_weight : float, optional
        The weight to give the BCE-based loss; the weight for the 
        dice-coefficient loss is always `1 - bce_weight`. The default is `0.5`.
    reweight : boolean, optional
        Whether to reweight the classes by calculating the BCE for each class
        then calculating the mean across classes. If `False`, then the raw BCE
        across all pixels, classes, and batches is returned (the default).
    class_weights: sequence, optional
        If provided, this list of weights allows the user to downweight /
        upweight different prediction classes' importance in the loss. E.g.,
        if class_weights = [w0, w1, w2], the three classes will be weighted
        with L_total = L_class0 * w0 + L_class1 * w1 + L_class2 * w2.
    smoothing : number, optional
        The smoothing coefficient `s` to use with the dice-coefficient liss.
        The default is `1`.
    metrics : dict or None, optional
        An optional dictionary in which the keys `'bce'`, `'dice'`, and `'loss'`
        should be mapped to floating-point values representing the cumulative
        losses so far across samples in the epoch. The losses of this
        calculation are added to these values.

    Returns
    -------
    number
        The weighted sum of losses of the prediction.
    """
    if bce_weight < 0 or bce_weight > 1:
        raise ValueError("bce_weight must be between 0 and 1")
    else:
        dice_weight = 1 - bce_weight
    if logits is None: logits = is_logits(pred)
    bce = bce_loss(pred, gold,
                   logits=logits,
                   reweight=reweight,
                   class_weights=class_weights,
                   metrics=metrics)
    dice = dice_loss(pred, gold,
                     logits=logits,
                     smoothing=smoothing,
                     class_weights=class_weights,
                     metrics=metrics)
    loss = bce * bce_weight + dice * dice_weight
    if metrics is not None:
        if 'loss' not in metrics: metrics['loss'] = 0.0
        metrics['loss'] += loss.data.cpu().numpy() * gold.size(0)
    return loss


# --- POSTPROCESSING (PIXEL <-> VERTEX) UTILITIES --- #

# data functions

def load_pickle(fp):
    """ Load single pickle file.
    """
    import pickle
    with open(fp, 'rb') as f:
        p = pickle.load(f)
    return p


def load_numpy(fp):
    """ Load single numpy file.
    """
    import numpy as np
    with open(fp, 'rb') as f:
        n = np.load(f)
    return n


def load_pickle_or_numpy_batch(
    fnames,
    prepend_fp=None,
    numpy=False
):
    """ Load a series of pickle or saved numpy files
        into a list, and return the list.
    """

    if prepend_fp is not None:
        fps = [os.path.join(prepend_fp, n) for n in fnames]
    else:
        fps = fnames

    load = load_numpy if numpy else load_pickle

    return [load(fp) for fp in fps]


# vertex functions

def px2vx(px_np, px2vx_dict, vcoords, ignore=None):
    """ Translates an array of pixel-values to a statistical map
        that lives on a cortical surface, according to
        a supplied dictionary.
        Arguments:
            px_np: Array of pixel values of shape (xdim, ydim, ...)
            px2vx_dict: Dictionary whose keys are (i, j) pixel locations
                and whose values are lists of vertices corresponding
                to a given pixel in a given view of a cortex.
            vcoords: Coordinates of the cortical mesh
            ignore: Value not to include in arrays.
        Returns:
            The vmap (cortex) version of the supplied px_np array
            of pixel values.
    """
    vlen = vcoords.shape[0]
    vmap_shape = tuple([vlen] + list(px_np.shape[2:]))
    vmap = np.zeros(vmap_shape)
    n1 = px_np.shape[0]
    n2 = px_np.shape[1]
    for i in range(n1):
        for j in range(n2):
            v = px2vx_dict[(i, j)]
            if v != ignore:
                vmap[v] = px_np[i, j]
    return vmap


def make_mask_vmap(a, b, threshold_value=2.0, method=None):
    """ Make a mask (boolean) vmap based on which values
        are below an allowed maximum threshold of difference
        between pxcoords and vxcoords.
        Arguments:
            a, b: The two arrays to compare, of same shape
                (n_entries, n_coordinates). Distances are calculated
                on a sum over the coordinates (axis=1).
            threshold_value: the maximum theshold, a positive float
            method: how to calculate the distance to threshold,
                by default, absolute value. Can also be 'square'
                to calculate squared distance.
        Returns: a boolean array in the shape of a and b
    """
    fn = np.abs
    if method == 'square':
        fn = np.square

    diff_vmap = (fn(a - b)).sum(axis=1)
    mask_vmap = (diff_vmap > threshold_value)

    return mask_vmap


# analysis functions

def dice(target, prediction, target_label, pred_label=None):
    """ Calculates DICE by comparing target to prediction.
        'label' is the value of the label to compare.
    """
    if pred_label is None:
        pred_label = target_label
    lt = target == target_label
    lp = prediction == pred_label

    tp = sum(lt * lp)
    fp = sum(lp) - tp
    fn = sum(lt) - tp
    dice = (2 * tp) / (2 * tp + fp + fn)

    return dice


def average_dice(target, prediction, labels, return_all=False):
    dices = [dice(target, prediction, l) for l in labels]
    if return_all:
        return np.mean(dices), dices
    else:
        return np.mean(dices)


# model prediction utilities
def model_eval(
    model,
    dl,
    df_metadata,
    df_idxs,
    save_dir,
    split_size=35,

):
    """
        Do a forward pass on a large dataset with multiple samples
        per subject EID. Iterate through and save predictions (batched by EID)
        as .npy files.
    """
    di = iter(dl)
    model.eval()
    device = get_device()

    for batch, (xs, ys) in enumerate(di):
        df = df_metadata.iloc[df_idxs[batch]]
        eid = str(df.iloc[0]['EID'])

        x1 = xs[:split_size]
        oversized = xs.shape[0] > split_size
        pred1 = model(x1.to(device))
        pred1 = pred1.cpu().detach().numpy()

        if oversized:
            x2 = xs[split_size:]
            pred2 = model(x2.to(device))
            pred2 = pred2.cpu().detach().numpy()
            pred = np.concatenate([pred1, pred2])
        else:
            pred = pred1

        save_pred_fn = f'{save_dir}/pred_02_{eid}.npy'
        with open(save_pred_fn, 'wb') as f:
            np.save(f, pred)
        print(f'Saved eid: {eid}')

        del xs
        del ys
        del pred1
        if oversized:
            del pred2
        del pred


def batch_px2v(
    df,
    ind,
    batch,
    load_dir,
    vcrd_dir,
    px2v_dir,
    anat_base_dir,
    anat_surf_subdir,
    anat_parc_subdir=None,
    save_dir=None,
    load_parc=False,
    preds_from_file=True
):
    """
        Converts a batch of predictions (grouped by subject)
        into a collection of surface vmaps.
        To achieve this, loads surface mesh, 2d pixelwise predictions, etc.
    """

    df_batch = df.iloc[ind[batch]]

    # collect batch metadata from df
    px2v_colname = 'Filename_v'
    pcrd_colname = 'Filename_p'
    px2v_fns = list(df_batch[px2v_colname])
    pcrd_fns = list(df_batch[pcrd_colname])
    eid = str(df_batch['EID'].iloc[0])

    # load px2v maps
    vcrd_batch = load_pickle_or_numpy_batch(pcrd_fns, vcrd_dir, numpy=True)
    px2v_batch = load_pickle_or_numpy_batch(px2v_fns, px2v_dir)

    # load freesurfer data
    mesh_fn = 'lh.inflated'
    curv_fn = 'lh.curv'

    mesh_fp = os.path.join(anat_base_dir, eid, anat_surf_subdir, mesh_fn)
    curv_fp = os.path.join(anat_base_dir, eid, anat_surf_subdir, curv_fn)

    mesh = surface.load_surf_mesh(mesh_fp)
    curv = surface.load_surf_data(curv_fp)

    # optional extra data for loading parc, if desired
    if load_parc:
        parc_fn = DESTRIEUX_FILENAME
        parc_fp = os.path.join(anat_base_dir, eid, anat_parc_subdir, parc_fn)

        parc = surface.load_surf_data(parc_fp)
        parc_selected = pu.get_filtered_parc(parc, fill_value=0.0)

    if preds_from_file:
        with open(f'{load_dir}/pred_02_{eid}.npy', 'rb') as f:
            preds = np.load(f)
    else:
        raise NotImplemented

    ypreds = torch.Tensor(preds)
    ypreds_softmax = F.softmax(ypreds, dim=1)

    # reshape predictions and move to numpy
    ypreds_np = ypreds.cpu().detach().numpy()
    # repeat for softmax
    ypreds_softmax_np = ypreds_softmax.cpu().detach().numpy()
    # place channel dim last
    ypreds_softmax_np = np.transpose(ypreds_softmax_np, axes=(0, 2, 3, 1))
    # eliminate channel dim by taking argmax
    ypreds_np_argmax = np.argmax(ypreds_np, axis=1)

    pred_vmaps = []
    prob_vmaps = []

    for i in range(preds.shape[0]):
        ypred_vmap = px2vx(ypreds_np_argmax[i],
                           px2v_batch[i],
                           mesh.coordinates)

        yprob_vmap = px2vx(ypreds_softmax_np[i],
                           px2v_batch[i],
                           mesh.coordinates)

        vcrd_vmap = px2vx(vcrd_batch[i],
                          px2v_batch[i],
                          mesh.coordinates)

        # mask out any dorsal predictions
        mask = make_mask_vmap(vcrd_vmap, mesh.coordinates)

        ypred_masked_vmap = ypred_vmap[:]
        ypred_masked_vmap[mask] = 0.0

        yprob_masked_vmap = yprob_vmap[:]
        yprob_masked_vmap[mask] = 0.0

        pred_vmaps.append(ypred_masked_vmap)
        prob_vmaps.append(yprob_masked_vmap)

    pred_vmaps = np.stack(pred_vmaps)
    prob_vmaps = np.stack(prob_vmaps)

    # save
    if save_dir is not None:
        save_pred_vmap_fn = f'vmap_pred_{eid}.npy'
        save_prob_vmap_fn = f'vmap_prob_{eid}.npy'

        with open(f'{save_dir}/{save_pred_vmap_fn}', 'wb') as f:
            np.save(f, pred_vmaps)
        with open(f'{save_dir}/{save_prob_vmap_fn}', 'wb') as f:
            np.save(f, prob_vmaps)

    # make class sums
    nclasses = 5
    classes = list(range(nclasses))
    class_sums = []
    for c in classes:
        class_sums.append(np.sum(pred_vmaps == c, axis=0))

    class_sums = np.stack(class_sums)

    if save_dir is not None:
        class_sum_fn = f'class_sums_{eid}.npy'
        with open(f'{save_dir}/{class_sum_fn}', 'wb') as f:
            np.save(f, class_sums)

    ret_dict = {}
    ret_dict['mesh'] = mesh
    ret_dict['curv'] = curv
    ret_dict['pred'] = pred_vmaps
    ret_dict['prob'] = prob_vmaps
    ret_dict['df'] = df_batch

    return ret_dict


def load_vmaps(
    eid,
    anat_dir,
    vmap_dir,
    mesh_subdir='surf/lh.inflated',
    data_subdir='surf/lh.curv',
    vmap_prefix='vmap_prob_'
):
    """
        Loads a subject's cortical mesh, curv file, and vmap.
        Returns as a dictionary.
    """
    mesh_fp = f'{anat_dir}/{eid}/{mesh_subdir}'
    surf_fp = f'{anat_dir}/{eid}/{data_subdir}'
    vmap_fp = f'{vmap_dir}/{vmap_prefix}{eid}.npy'

    mesh = surface.load_surf_mesh(mesh_fp)
    curv = surface.load_surf_data(surf_fp)
    vmap = load_numpy(vmap_fp)

    retn = {
        'mesh': mesh,
        'curv': curv,
        'vmap': vmap
    }

    return retn


def segment_ots(
    eid,
    anat_dir,
    vmap_dir,
    min_size=50,
    prob_threshold=float(4 / 7),
    extract_label=2,
    mesh_subdir='surf/lh.inflated',
    data_subdir='surf/lh.curv',
    vmap_prefix='vmap_prob_'
):
    """
        Load a subjects saved probability maps.
        Extract features of the OTS.
    """
    sub_data = load_vmaps(
        eid,
        anat_dir,
        vmap_dir,
        mesh_subdir=mesh_subdir,
        data_subdir=data_subdir,
        vmap_prefix=vmap_prefix
    )
    mesh = sub_data['mesh']
    curv = sub_data['curv']
    vmap = sub_data['vmap']

    tri = trimesh.Trimesh(
        vertices=mesh.coordinates,
        faces=mesh.faces
    )

    prob = vmap.mean(axis=0)[:, extract_label]
    idxs = np.array(range(len(prob)))
    idxs = idxs[prob > prob_threshold]
    mask_edges = []

    # get only edges in label of interest
    for e in tri.edges:
        a, b = e
        if a in idxs or b in idxs:
            mask_edges.append(e)

    mask = np.stack(mask_edges)
    ccs = trimesh.graph.connected_components(mask)  # list<vertices> of conn. components
    ccs = [cc for cc in ccs if cc.shape[0] >= min_size]  # filter out very small components
    cc_coordinates = [mesh.coordinates[c] for c in ccs]
    ranges = [[max(cc[:, 1]), min(cc[:, 1])] for cc in cc_coordinates]
    sizes = [cc.shape[0] for cc in ccs]

    return sizes, ranges, ccs


# ==============================================================================
# __all__

__all__ = ["is_partition"
           ,"trndata"
           ,"valdata"
#           ,"partition"
           ,"partition_id"
           ,"kernel_default_padding"
           ,"convrelu"
           ,"is_logits"
           ,"dice_loss"
           ,"bce_loss"
           ,"loss"]
