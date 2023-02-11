import os
import sys

import numpy as np
import pandas as pd

from nilearn import surface

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

from PIL import Image
from PIL import ImageOps

import PIL
from PIL import Image
from PIL import ImageOps
# define default matplotlib palette
from matplotlib import cm
from matplotlib import colors


### GLOBAL VARIABLES ###

# coordinate ranges
RANGES = [60.0, 150.0, 100.0]
DESTRIEUX_LABELS = [2, 19, 21, 23, 24, 25, 30, 37, 38, 50,
                    51, 57, 58, 59, 60, 61, 63, 65]
DESTRIEUX_FILENAME = 'lh.aparc.a2009s.annot'

# default palette
tab20 = cm.get_cmap('tab20')
tab20_colors = 255 * np.array(tab20.colors)
PAL = np.concatenate(([[255., 255., 255.]], tab20_colors))

# alternate palette for OTS labels
OTS_CMAP = colors.ListedColormap(['blue',
                                  'green',
                                  'yellow',
                                  'red',
                                  'purple',
                                  'pink'])
ots_colors = np.array([colors.to_rgb(c) for c in OTS_CMAP.colors])
PAL_OTS = np.concatenate(([[1.0, 1.0, 1.0]], ots_colors))

# alternate palette for OTS-MFS-GAP labels
OTS_MFS_GAP_CMAP = colors.ListedColormap(['blue',
                                          'green',
                                          'yellow',
                                          'red',
                                          'purple',
                                          'pink',
                                          'brown',
                                          'orange']
                                         )
ots_mfs_gap_colors = np.array([colors.to_rgb(c)
                               for c in OTS_MFS_GAP_CMAP.colors])
PAL_OTS_MFS_GAP = np.concatenate(([[1.0, 1.0, 1.0]], ots_mfs_gap_colors))


def make_random_angles(n, center, scale):
    """ Generates n random angles about with certain scale
        centered on center. Angles are drawn from random
        uniform distribution, center +/- scale.
    """
    c = np.array(center)
    return c + (2.0 * scale) * (np.random.rand(n, 2) - 0.5)


def make_angles(ntotal, center=[210.0, 90.0], scale=15.0,
                nonrandom_angles=None):
    """ Generates angles using (elev, azim) format.
        Generates n random angles about with certain scale
        centered on center.

        Some are drawn from random
        uniform distribution, center +/- scale.

        Optionally, a list of pre-specified (non-random)
        angles may be supplied, in which case
        ntotal - len(nonrandom_angles) random angles are generated
        so that the total number of angles returned is ntotal.
    """
    if nonrandom_angles is not None:
        n = ntotal - len(nonrandom_angles)
        assert n >= 0
        nonrandom = np.array(nonrandom_angles)
        assert nonrandom.shape[1] == 2  # angles are specified as (elev, azim)
    else:
        n = ntotal
    random_angles = make_random_angles(n, center=center, scale=scale)

    angles = None
    if nonrandom_angles is not None:
        angles = np.concatenate((nonrandom, random_angles))
    else:
        angles = random_angles
    return angles


def get_random_shift(px, fraction=0.1):
    shift_max = int(fraction * px)
    return np.random.randint(low=-shift_max, high=shift_max, size=(2,))


### Helper functions for shifts (translations) ###

# same parameters for all images
def crop(image):
    """ Cropping helper function.
        Assumes input has correct y dimensions and
        x dim >= y dim; trims excess x dim to make a square.
    """
    x0, y0 = image.size
    diff = (x0 - y0) // 2
    xl, xr = diff, x0 - diff
    ltrb = (xl, 0, xr, y0)
    return image.crop(ltrb)


def resize(image, newsize=256):
    return image.resize((newsize, newsize))


def get_closest_color(col, pal):
    return np.abs(col - pal).sum(axis=1).argmin()


def to_channel_img(img, pal):
    """ Converts an image into an image with one scalar value
        for each color in a specified palette. Each pixel's
        value is determined to be the index (in the palette)
        of the color that is closest to the pixel's original
        value.

        Note: expects image in RGBA PNG format, e.g. image
        shape should be (x_len, y_len, 4)).
    """
    s = img.shape
    ret_img = np.zeros(s[:-1])

    # loop over pixels
    for x in range(s[0]):
        for y in range(s[1]):
            px_color = img[x, y]
            cc = get_closest_color(px_color, pal)
            ret_img[x, y] = cc
    return ret_img


def np_onehot(x, n=None):
    """ Reshapes an array of integers into a one-hot-encoded
        vector of 0's and 1's by adding an extra dimension.
        Not in-place; returns the onehot np array.
    """
    if n is None:
        n = x.max() + 1
    return np.eye(n)[x.astype(int)]


# helper for parsing mask images
def split(array, threshold=70, bg=255):
    T = threshold
    t = array.copy()

    t[array < T] = 2.0
    t[array >= T] = 1.0
    t[array == bg] = 0.0

    return t


def fig_to_PIL(fig):
    """ Converts a matplotlib figure into a PIL figure object.
    """
    from PIL import Image

    fig.canvas.draw()
    figstring = fig.canvas.tostring_rgb()
    data = np.frombuffer(figstring, dtype=np.uint8)
    pdim = np.sqrt(data.shape[0] // 3).astype(int)  # 3 for RGB

    return Image.frombytes('RGB', (pdim, pdim), figstring)


def process(image, newsize=256, grayscale=True):
    """Crops image, resizes, and converts to grayscale."""
    image = crop(image)
    image = resize(image, newsize)
    if grayscale:
        image = ImageOps.grayscale(image)
    return image


# function to process parc images into np files
def process_parc_img(img, pal=None, img_out_fp=None, newsize=256, shifts=None,
                     resample=None, normalize=True):
    """ Downsamples an image, converts it to a numpy array,
        resamples each pixel color to nearest color
        in a specified palette (pal),
        and saves the result as a numpy array.
        Arguments:
            img: a PIL image
            pal: palette to resample to (np array of shape (nclrs, 3))
            img_out_fp: if None, output is not saved
            newsize: size to resize to (image should be square)
            resample: (optional) resampling algorithm to use during image
                resizing. Default (None) leads to PIL.Image.NEAREST. If
                reample='default', then the default PIL resampling method
                is used instead.
            normalize: (Boolean, default=True, or else float value). If False,
                no normalization will be performed. If True, normalization
                value will be inferred to be the maximum value of the image.
                If the value is manually provided, this is used instead.
        Returns:
            the resulting np array of shape (newsize, newsize,)
            with values in the range 0, pal.shape[0] - 1
    """
    if pal is None:
        palette = PAL
    elif pal == 'OTS':
        palette = PAL_OTS
    elif pal == 'OTS-MFS-GAP':
        palette = PAL_OTS_MFS_GAP
    else:
        palette = pal

    # downsampling logic
    if resample is None:
        resample = PIL.Image.NEAREST
    elif resample == 'default':
        resample = None

    img = img.resize((newsize, newsize), resample=resample)
    img = np.array(img)

    # parse normalize logic
    if isinstance(normalize, bool):
        if normalize:
            norm_value = np.max(img)
        else:
            norm_value = 1.0
    else:
        if not isinstance(normalize, float):
            raise TypeError
        norm_value = normalize
    img = img / norm_value
    img_channel = to_channel_img(img, pal=palette)

    if shifts is not None:
        img_channel = shift_single(img_channel, shifts=shifts)

    # save (optionally)
    if img_out_fp is not None:
        np.save(img_out_fp, img_channel)

    return img_channel


# function to process curvature images into np files
def process_curv_img(img, newsize=256, img_out_fp=None, shifts=None):
    """Crops image, resizes, and converts to grayscale."""
    img = img.resize((newsize, newsize))
    img = ImageOps.grayscale(img)
    img_np = np.array(img)

    if shifts is not None:
        img_np = shift_single(img_np, shifts=shifts)

    if img_out_fp is not None:
        np.save(img_out_fp, img_np)
    return img_np


def process_mask_img(img, img_out_fp=None, newsize=256, shifts=None):
    """ Downsamples an image, converts it to a numpy array,
        resamples each pixel color to nearest color
        in a specified palette (pal),
        and saves the result as a numpy array.
        Arguments:
            img_fp: full filepath to an RGBA image
            pal: palette to resample to (np array of shape (nclrs, 3))
            img_out_fp: if None, output is not saved
            newsize: size to resize to (image should be square)
        Returns:
            the resulting np array of shape (newsize, newsize,)
            with values in the range 0, pal.shape[0] - 1
    """
    img = ImageOps.grayscale(img)
    img = img.resize((newsize, newsize))
    img = np.array(img)
    img_channel = split(img)

    if shifts is not None:
        img_channel = shift_single(img_channel, shifts=shifts)

    # save (optionally)
    if img_out_fp is not None:
        np.save(img_out_fp, img_channel)

    return img_channel


### utilities for translating images ###

def shift(xin, yin, fraction=0.1, shifts=None, channel=False, returnparams=True):
    """ Generate random shifts (translations / affine transformations)
        in np arrays that represent images.
        Assumes images are square. Can have optional channel in dim 0.
    """
    assert fraction < 1.0
    px = xin.shape[1]

    if shifts is None:
        shift_max = int(fraction * px)
        shifts = np.random.randint(low=-shift_max, high=shift_max, size=(2,))
    s1, s2 = shifts
    left = s1 < 0
    up = s2 < 0
    right = not left
    down = not up

    new_x = np.zeros_like(xin)
    new_y = np.zeros_like(yin)

    if channel:

        if left and up:
            new_x[:, 0:px + s1, 0:px + s2] = xin[:, np.abs(s1):, np.abs(s2):]
            new_y[:, 0:px + s1, 0:px + s2] = yin[:, np.abs(s1):, np.abs(s2):]
        elif left and down:
            new_x[:, 0:px + s1, s2:] = xin[:, np.abs(s1):, 0:px - s2]
            new_y[:, 0:px + s1, s2:] = yin[:, np.abs(s1):, 0:px - s2]
        elif right and up:
            new_x[:, s1:, 0:px + s2] = xin[:, 0:px - s1, np.abs(s2):]
            new_y[:, s1:, 0:px + s2] = yin[:, 0:px - s1, np.abs(s2):]
        else:
            new_x[:, s1:, s2:] = xin[:, 0:px - s1, 0:px - s2]
            new_y[:, s1:, s2:] = yin[:, 0:px - s1, 0:px - s2]

    else:  # no channel dimension

        px = xin.shape[0]
        if left and up:
            new_x[0:px + s1, 0:px + s2] = xin[np.abs(s1):, np.abs(s2):]
            new_y[0:px + s1, 0:px + s2] = yin[np.abs(s1):, np.abs(s2):]
        elif left and down:
            new_x[0:px + s1, s2:] = xin[np.abs(s1):, 0:px - s2]
            new_y[0:px + s1, s2:] = yin[np.abs(s1):, 0:px - s2]
        elif right and up:
            new_x[s1:, 0:px + s2] = xin[0:px - s1, np.abs(s2):]
            new_y[s1:, 0:px + s2] = yin[0:px - s1, np.abs(s2):]
        else:
            new_x[s1:, s2:] = xin[0:px - s1, 0:px - s2]
            new_y[s1:, s2:] = yin[0:px - s1, 0:px - s2]

    if returnparams:
        return new_x, new_y, s1, s2
    else:
        return new_x, new_y


def shift_from_file(xin_fp, yin_fp, xout_path, yout_path, niter=1, channel=False):
    """ Load .npy arrays from filepaths (including name)
        and save randomly translated versions of the same arrays in corresponding
        out directories; shift parameters will be part of save filename.
    """
    xin = np.load(xin_fp)
    yin = np.load(yin_fp)

    for i in range(niter):
        xshift, yshift, s1, s2 = shift(xin, yin, shifts=(0, 0), channel=channel)
        x_fname = (xin_fp.split('/')[-1])[:-4]
        y_fname = (yin_fp.split('/')[-1])[:-4]
        np.save(f'{xout_path}/{x_fname}-trans-{s1}-{s2}', xshift)
        np.save(f'{yout_path}/{y_fname}-trans-{s1}-{s2}', yshift)
    return


def shift_single(xin, fraction=0.1, shifts=None, channel=False, returnparams=True):
    """ Generate random shifts (translations / affine transformations)
        in np arrays that represent images.
        Assumes images are square. Can have optional channel in dim 0.
    """
    assert fraction < 1.0
    px = xin.shape[1]

    if shifts is None:
        shift_max = int(fraction * px)
        shifts = np.random.randint(low=-shift_max, high=shift_max, size=(2,))
    s1, s2 = shifts
    left = s1 < 0
    up = s2 < 0
    right = not left
    down = not up

    new_x = np.zeros_like(xin)

    if channel:

        if left and up:
            new_x[:, 0:px + s1, 0:px + s2] = xin[:, np.abs(s1):, np.abs(s2):]
        elif left and down:
            new_x[:, 0:px + s1, s2:] = xin[:, np.abs(s1):, 0:px - s2]
        elif right and up:
            new_x[:, s1:, 0:px + s2] = xin[:, 0:px - s1, np.abs(s2):]
        else:
            new_x[:, s1:, s2:] = xin[:, 0:px - s1, 0:px - s2]

    else:  # no channel dimension

        px = xin.shape[0]
        if left and up:
            new_x[0:px + s1, 0:px + s2] = xin[np.abs(s1):, np.abs(s2):]
        elif left and down:
            new_x[0:px + s1, s2:] = xin[np.abs(s1):, 0:px - s2]
        elif right and up:
            new_x[s1:, 0:px + s2] = xin[0:px - s1, np.abs(s2):]
        else:
            new_x[s1:, s2:] = xin[0:px - s1, 0:px - s2]

    if returnparams:
        return new_x, s1, s2
    else:
        return new_x

### channel stitching utilities ###


def stitch_two_nps(np1, np2, second_to_onehot=True, n=None):
    if second_to_onehot:
        np2 = np_onehot(np2, n=n)
    np1 = np.expand_dims(np1, axis=2)
    np3 = np.concatenate((np1, np2), axis=2)
    return np3


def stitch_nps(nps, onehot_indices=None, n=None):
    """
        Stitches a list of np files together along
        a last axis, optionally encoding some dimensions
        as onehot vectors.
        Arguments:
            nps: list<np.array>, the arrays to concatenate along
                a new axis
            onehot_indices: (optional) list<int>, the indices of entries
                in nps that specify which elements to convert to one-hot
            n: number of one-hot dimensions (classes) if consistent across
                inputs.
        Returns:
            The input arrays joined into a single new array as channels.
    """
    if onehot_indices is None:
        onehot_indices = []

    for i, arr in enumerate(nps):
        if i in onehot_indices:
            nps[i] = np_onehot(nps[i], n=n)
        else:
            nps[i] = np.expand_dims(arr, axis=2)
    nps_concat = np.concatenate(tuple(nps), axis=2)

    return nps_concat


def stitch_files(f1, f2, save_fname, second_to_onehot=True, n=None,
                 return_array=False):
    np1 = np.load(f1)
    np2 = np.load(f2)
    np3 = stitch_two_nps(np1, np2, second_to_onehot=second_to_onehot, n=n)
    with open(save_fname, 'wb') as f:
        np.save(f, np3)
    if return_array:
        return np3


def stitch_channels(np_dict, key_list, return_key_name='np_x', inplace=True):
    """ Stitches the channels listed in key_list
        into a single, larger array along the
        channel dimension.
        For inputs of the form dict<list<np.array>>,
        where the list represents different views of a subject,
        this yields a dict with a new entry np_x.
        If inplace is True, then the new entry is added to the original dictionary
        np_dict.
    """
    import copy
    assert all(k in np_dict.keys() for k in ['angles'] + key_list)
    angles = np_dict['angles']

    np_xs = []
    for i, a in enumerate(angles):
        np_curv = np_dict['curv'][i]
        np_mask = np_dict['mask'][i]

        np_x = stitch_nps(np_curv, np_mask)  # create single np array
        np_xs.append(np_x)

    if inplace:
        ret_dict = np_dict
    else:
        ret_dict = copy.deepcopy(np_dict)

    ret_dict[return_key_name] = np_xs
    return ret_dict


### utilities for creating px2vertex maps ###


def get_coords(arr, vrange, img_max=255.0):
    """ Convert color value (default: 0 - 255) into coordinate value.
    """
    img_max = float(img_max)  # in case img_max is supplied as int
    arr = arr.astype(dtype=np.float32)
    arr[arr == img_max] = np.nan  # set max (white background to nan)
    arr *= (2 * vrange / img_max)
    arr -= vrange
    return arr


def get_pixel_coords(np_x, np_y, np_z, xrange, yrange, zrange, img_max=255.0):
    """ Convert color into coordinate, once for each coordinate (x, y, z)
    """
    xc = get_coords(np_x, xrange)
    yc = get_coords(np_y, yrange)
    zc = get_coords(np_z, zrange)
    return np.stack([xc, yc, zc], axis=2)


def coords_to_vertices(pixel_coords, vertex_coords, return_distances=False):
    """ Loops over all vertices to match them to their closest pixels.
        Arguments:
            pixel_coords: the best approximation to the vertex coordinate
                visualized in the input images of interest
            vertex_coords: an np array of shape (V, 3)
                containing the coordinates of the cortical mesh of interest
            return_distances: (optional boolean, default=False)
                whether to return the distances between vertex and closest pixel
                found during this reconstruction process.
        Returns:
            px2vert_dict: a dictionary with keys (i, j) pixels in image of interest,
                vales [v1, v2, ...] the vertices whose closest pixel is (i, j).
            (optional) dist_list: the list of distances returned if
                return_distances == True
    """
    from sklearn.neighbors import KDTree

    # reshape px coordinates, remove nans
    px_shape = pixel_coords.shape[:2]
    pcs_flat = pixel_coords.reshape(-1, 3)
    pcs_flat_isnan = np.any(np.isnan(pcs_flat), axis=1)
    pcs_flat_nonan = pcs_flat[~pcs_flat_isnan]

    px_indices = np.array([(i, j) for i in range(px_shape[0]) for j in range(px_shape[1])])
    px_indices_filtered = px_indices[~pcs_flat_isnan]

    # create tree and query it
    ptree = KDTree(pcs_flat_nonan)
    dist_list, closest_pxs = ptree.query(vertex_coords)

    # reshape closest_pxs and refill nan's
    vxpx = np.take(px_indices_filtered, closest_pxs, axis=0).squeeze()

    # return results as dictionary
    # keys: (i, j) of pixels, values: list of vertices
    px2v_dict = {}
    for i in range(px_shape[0]):
        for j in range(px_shape[1]):
            px2v_dict[(i, j)] = []
    for v, px in enumerate(vxpx):
        (i, j) = px
        px2v_dict[(i, j)].append(v)

    if return_distances:
        return px2v_dict, dist_list
    else:
        return px2v_dict


def px_2_coord(vcoords, pcoords, px_vertex_dict, distance_dict=False):
    coords_vert_recon = np.zeros_like(vcoords)
    for k, v in px_vertex_dict.items():
        i, j = k  # unpack k
        coord = pcoords[i, j, :]
        for vertex in v:
            if distance_dict:
                vertex = vertex[0]
            coords_vert_recon[vertex] = coord
    return coords_vert_recon


def process_coord_img(img, ns=256):
    """ Resizes PIL image and converts to numpy.
        Args:
            img: PIL image object
            ns: (int, optional) new size in pixels.
    """
    img = ImageOps.grayscale(img)
    return np.array(img.resize((ns, ns)))


def make_pixel_map(xnp, ynp, znp, surf_coords, angle, ns=256,
                   shift=None, alpha=False, ranges=None):
    """ Make pixel maps from specified images.
        Arguments:
            ximg: PIL image of Freesurfer x-coordinate visualized on cortex.
            yimg: PIL image of Freesurfer y-coordinate visualized on cortex.
            zimg: PIL image of Freesurfer z-coordinate visualized on cortex.
            (Note: all images must be from the same view.)
            angle: a sequence (elev, azim) specifying the view angle of
                ximg, yimg, and zimg.
            surf_coords: an np array of shape (V, 3) containing the data
                of each x, y, z coordinate for every vertex in the
                cortical mesh being visualized in ximg, yimg, zimg.
            ns: (Optional, int) the size to which to downsample the resulting image.
                If None, defaults to 256.
            shift: (Optional, (int, int)) A pair of integers specifying a shift
                (in number of pixels) to apply in the down, right directions
                to the downsampled image.
            ranges: (Optional) list<float>, a list of three floats representing
                the +/- deviations from 0 that bound x, y, and z coordinates,
                used to create the original x, y, z coordinate images of the
                cortex in question.
        Returns:
            A tuple (pcs_down, vx2pxdown_dict)
                px_coords: the coordinates of each nearest pixel

    """
    if ranges is None:
        ranges = RANGES

    xrange = ranges[0]
    yrange = ranges[1]
    zrange = ranges[2]

    # note: shift image pixels here
    if shift is not None:
        xnp = shift_single(xnp, shifts=shift, returnparams=False)
        ynp = shift_single(ynp, shifts=shift, returnparams=False)
        znp = shift_single(znp, shifts=shift, returnparams=False)

    px_coords = get_pixel_coords(xnp, ynp, znp, xrange, yrange, zrange)
    px2v_dict = coords_to_vertices(px_coords, surf_coords)

    return px_coords, px2v_dict


def get_filtered_parc(parc, mode=None, fill_value=-1):
    """ Filter parcellation by including only certain labels.
        In default mode, selects only the parcellations of Destrieux
        that are adjacent to the OTS.
    """
    if mode is None:
        selected_parc = np.array([DESTRIEUX_LABELS.index(l)
                                  if l in DESTRIEUX_LABELS
                                  else fill_value
                                  for l in parc]
                                 )
    else:
        selected_parc = parc
    return selected_parc


def get_freesurfer_subject(sub_dir, mesh_files, data_files,
                           surf_dir='surf'):
    from nilearn import surface
    import os
    """ Given a subject's freesurfer directory, loads
        each freesurfer file in both mesh_files and map_files, and returns
        their nilearn objects in a dictionary {'name': nilearn_object}.
        Arguments:
            sub_dir: (str) filepath to subject's top-level directory,
                which would normally contain, e.g. /surf, /label, ...
            mesh_files: (list<str>) names of cortical meshes to load
            data_files: (list<str>) names of cortical stat_map files to load
            surf_dir: (str) subdirectory of sub_dir that contains freesurfer
                files of interest. Defaults to 'surf'.
        Returns:
            a dictionary {'name': nilearn_object} of each loaded object; names
            are the same as specified in mesh_files and map_files.
    """
    cortex_data = {}

    if mesh_files is not None:
        for mf in mesh_files:
            fp = os.path.join(sub_dir, surf_dir, mf)
            cortex_data[mf] = surface.load_surf_mesh(fp)

    if data_files is not None:
        for df in data_files:
            fp = os.path.join(sub_dir, surf_dir, df)
            cortex_data[df] = surface.load_surf_data(fp)

    return cortex_data


def freesurfer_subject_data_exists(sub_dir, mesh_files, data_files,
                                   surf_dir='surf'):
    from nilearn import surface
    import os
    """ Given a subject's freesurfer directory, checks all
        freesurfer files in both mesh_files and map_files, and returns
        True iff they all exist.
        Arguments:
            sub_dir: (str) filepath to subject's top-level directory,
                which would normally contain, e.g. /surf, /label, ...
            mesh_files: (list<str>) names of cortical meshes to load
            data_files: (list<str>) names of cortical stat_map files to load
            surf_dir: (str) subdirectory of sub_dir that contains freesurfer
                files of interest. Defaults to 'surf'.
        Returns:
            a boolean, True only if all required freesurfer subject files exist.
    """
    mesh_files_exist = []
    if mesh_files is not None:
        for mf in mesh_files:
            fp = os.path.join(sub_dir, surf_dir, mf)
            mesh_files_exist.append(os.path.exists(fp))

    data_files_exist = []
    if data_files is not None:
        for df in data_files:
            fp = os.path.join(sub_dir, surf_dir, df)
            data_files_exist.append(os.path.exists(fp))

    return all(mesh_files_exist + data_files_exist)


def get_freesurfer_subject_with_parc(sub_dir, mesh_files, data_files,
                                     surf_dir='surf',
                                     label_files=None,
                                     label_dir='label'):
    """ Given a subject's freesurfer directory, loads
        each freesurfer file in both mesh_files and map_files, and returns
        their nilearn objects in a dictionary {'name': nilearn_object}.
        Also loads label files from the surf_dir 'label'
        Arguments:
            sub_dir: (str) filepath to subject's top-level directory,
                which would normally contain, e.g. /surf, /label, ...
            mesh_files: (list<str>) names of cortical meshes to load
            data_files: (list<str>) names of cortical stat_map files to load
            surf_dir: (str) subdirectory of sub_dir that contains freesurfer
                files of interest. Defaults to 'surf'.
            label_files: list<str> names of label files to load from the /label dir.
                Defaults to Destrieux parcellation, lh.aparc.a2009s.annot
            label_dir: (str, optional) Subdirectory of subject's freesurfer directory
                from which to load label_files. Defaults to 'label/'. This
                parameter is useful when dealing with e.g., labels organized
                in subdirectories of label/, such as label/manual_label/
        Returns:
            a dictionary {'name': nilearn_object} of each loaded object; names
            are the same as specified in mesh_files and map_files.
    """
    if label_files is None:
        label_files = [DESTRIEUX_FILENAME]
    surf_dict = get_freesurfer_subject(sub_dir, mesh_files, data_files)
    label_dict = get_freesurfer_subject(sub_dir,
                                        mesh_files=None,
                                        data_files=label_files,
                                        surf_dir=label_dir)
    surf_dict.update(label_dict)
    return surf_dict


def freesurfer_subject_data_exists_parc(sub_dir, mesh_files, data_files,
                                        surf_dir='surf',
                                        label_files=None):
    """ Given a subject's freesurfer directory, loads
        each freesurfer file in both mesh_files and map_files, and returns
        their nilearn objects in a dictionary {'name': nilearn_object}.
        Also loads label files from the surf_dir 'label'
        Arguments:
            sub_dir: (str) filepath to subject's top-level directory,
                which would normally contain, e.g. /surf, /label, ...
            mesh_files: (list<str>) names of cortical meshes to load
            data_files: (list<str>) names of cortical stat_map files to load
            surf_dir: (str) subdirectory of sub_dir that contains freesurfer
                files of interest. Defaults to 'surf'.
            label_files: list<str> names of label files to load from the /label dir.
                Defaults to Destrieux parcellation, lh.aparc.a2009s.annot
        Returns:
            a dictionary {'name': nilearn_object} of each loaded object; names
            are the same as specified in mesh_files and map_files.
    """
    if label_files is None:
        label_files = [DESTRIEUX_FILENAME]
    surf_files_exist = freesurfer_subject_data_exists(sub_dir, mesh_files, data_files)
    label_files_exist = freesurfer_subject_data_exists(sub_dir, mesh_files=None,
                                                       data_files=label_files, surf_dir='label')
    return surf_files_exist and label_files_exist


def make_subject_coord_images(mesh, figs=None, a=None, figsize=(8, 8)):
    """ Make subject image to record x, y, and z of vertices.
        Saved as 3 grayscale images.
        # TODO: passing a matplotlib `figure` as an arg is not
        # currently supported.
    """
    from nilearn import plotting

    if figs is not None:
        raise NotImplementedError

    if a is None:
        a = [210.0, 90.0]

    coord_figs = []
    for i in range(3):  # one image per coordinate (x, y, z)

        fig, _ = plt.subplots(figsize=figsize)
        plotting.plot_surf_roi(mesh, mesh.coordinates[:, i]
                               , view=(a[0], a[1])
                               # ,bg_map=test_curv
                               # ,bg_on_data=True
                               , figure=fig
                               , cmap='gray'
                               # ,output_file=None
                               , vmin=-RANGES[i]
                               , vmax=RANGES[i]
                               # ,threshold=25.0
                               # colorbar=True
                               )
        coord_figs.append(fig)
    return coord_figs


def make_subject_images(mesh, curv, parc, extra_channels_dict=None,
                        parc_save_dir=None,
                        curv_save_dir=None,
                        angles=None,
                        nangles=20,
                        nonrandom_angles=None,
                        make_coord_figs=True,
                        make_mask_figs=True,
                        mode=None
                        ):
    """ Main image-generation function for UNet input images.
        Arguments:
            mesh: a nilearn cortical mesh object
            curv: the stat_map nilearn object corresponding to
                curvature on the above mesh.
            parc: the stat_map nilearn object corresponding to
                parcellation on the above mesh.
            extra_channels_dict: (optional) if specified, a dictionary of
                keys, values where the keys are names of stat maps to plot
                and values are the stat maps.
            nangles: the total number of angles to generate for this subject.
            nonrandom_angles: (optional) a sequence of angles to include
                that are not randomly generated upon calling this function.
            make_coord_figs: (boolean) if True, 3 angles (x, y, z)
                will be returned for each image
            mode: (string, optional) default is 'HBN', but can also be set
                to 'OTS' or to 'OTS-MFS-GAP'. This parameter sets the color
                palette to be used, which varies based on the type of input
                data.
        Returns:
            A dictionary of lists of figures, dict<list<fig>>:
                'parc': [parcellation figures]
                'curv': [(continuous) curvature figures]
                'mask': [(binary masked curvature figures)]
                'xcoord': [figures of x coordinates in grayscale]
                'ycoord': [figures of y coordinates in grayscale]
                'zcoord': [figures of z coordinates in grayscale]
                'angles': [angles used to generate figures, in same order]
                ... optionally, a list of extra keys, the same keys as
                given in extra_channels_dict, each containing a list of
                figures of the corresponding stat maps.
    """
    from nilearn import plotting

    FIGSIZE = (8, 8)
    DESTRIEUX_LABELS = [2, 19, 21, 23, 24, 25, 30, 37, 38, 50, 51, 57, 58, 59, 60, 61, 63, 65]
    if mode is None:
        selected_parc = np.array([DESTRIEUX_LABELS.index(l) if l in DESTRIEUX_LABELS else -1 for l in parc])
    else:
        selected_parc = parc

    parc_figs = []
    curv_figs = []
    mask_figs = []
    xcoord_figs = []
    ycoord_figs = []
    zcoord_figs = []
    if extra_channels_dict is not None:
        extra_channel_figs = {}
        for k, v in extra_channels_dict.items():
            extra_channel_figs[k] = []
    return_dict = {}

    if angles is None:
        angles = make_angles(nangles, nonrandom_angles=nonrandom_angles)

    for a in angles:

        # plot parcellation
        if mode is None:
            parc_fig, _ = plt.subplots(figsize=FIGSIZE)
            plotting.plot_surf_roi(mesh, selected_parc
                                   , view=(a[0], a[1])
                                   # ,bg_map=test_curv
                                   # ,bg_on_data=True
                                   , figure=parc_fig
                                   , cmap='tab20'
                                   # , output_file=parc_save_fp
                                   # ,threshold=25.0
                                   # colorbar=True
                                   )
            parc_figs.append(parc_fig)
        else:
            # OTS modes are similar; the MFS-GAP setting contains
            # two additional labels, hence uses a different palette.
            if mode == 'OTS':
                cmap = OTS_CMAP
                vmax = 6.0
            if mode == 'OTS-MFS-GAP':
                vmax = 8.0
                cmap = OTS_MFS_GAP_CMAP

            parc_fig, _ = plt.subplots(figsize=FIGSIZE)
            plotting.plot_surf_roi(mesh, selected_parc
                                   , view=(a[0], a[1])
                                   # ,bg_map=test_curv
                                   # ,bg_on_data=True
                                   , figure=parc_fig
                                   , cmap=cmap
                                   , vmin=1.0
                                   , vmax=vmax
                                   # , output_file=parc_save_fp
                                   # ,threshold=25.0
                                   # colorbar=True
                                   )
            parc_figs.append(parc_fig)


        # plot curvature
        above_parc_threshold = selected_parc.max() + 1.0  # set threshold to exclude parc
        curv_fig, _ = plt.subplots(figsize=FIGSIZE)
        plotting.plot_surf_roi(mesh, selected_parc
                               , view=(a[0], a[1])
                               , bg_map=curv
                               # , bg_on_data=True
                               , figure=curv_fig
                               , cmap='tab20'
                               , threshold=above_parc_threshold
                               # colorbar=True
                               )
        curv_figs.append(curv_fig)

        # plot mask figures if desired
        if make_mask_figs:
            curv_mask = np.zeros_like(curv)
            curv_mask[curv > 0] = 1.0

            mask_fig, _ = plt.subplots(figsize=FIGSIZE)
            plotting.plot_surf_roi(mesh, curv_mask
                                   , view=(a[0], a[1])
                                   # ,bg_on_data=True
                                   , figure=mask_fig
                                   , cmap='gray'
                                   , threshold=0.0
                                   # ,output_file=f'{out_dir}/{sub}-{a[0]:.2f}-{a[1]:.2f}-curv-mask.png'
                                   # ,colorbar=True
                                   )
            mask_figs.append(mask_fig)

        # plot x, y, z coordinates as grayscale images
        if make_coord_figs:
            x, y, z = make_subject_coord_images(mesh, a=a, figsize=FIGSIZE)
            xcoord_figs.append(x)
            ycoord_figs.append(y)
            zcoord_figs.append(z)

        if extra_channels_dict is not None:
            for key, statmap in extra_channels_dict.items():
                # plot statmaps and save under channel key
                above_parc_threshold = selected_parc.max() + 1.0  # set threshold to exclude parc
                channel_fig, _ = plt.subplots(figsize=FIGSIZE)
                plotting.plot_surf_roi(mesh, selected_parc
                                       , view=(a[0], a[1])
                                       , bg_map=statmap
                                       # ,bg_on_data=True
                                       , figure=channel_fig
                                       , cmap='tab20'
                                       , threshold=above_parc_threshold
                                       # colorbar=True
                                       )
                extra_channel_figs[key].append(channel_fig)

    # package returns into single dictionary
    return_dict['mask'] = mask_figs
    return_dict['curv'] = curv_figs
    return_dict['parc'] = parc_figs
    return_dict['xcoord'] = xcoord_figs
    return_dict['ycoord'] = ycoord_figs
    return_dict['zcoord'] = zcoord_figs
    return_dict['angles'] = angles

    # add extra channels
    if extra_channels_dict is not None:
        for key, value in extra_channel_figs.items():
            return_dict[key] = value

    return return_dict


def process_figs(img_dict, ns=256, mode=None, resample=None):
    """ Processes images in preparation for UNet.
        Several successive steps are applied.
        1. Downsampling
        2. Conversion to numpy
        3. Parsing colors to int channels from 0, 1, ... nchannels - 1
        Args:
            img_dict: (dict<list<fig>>) a dictionary of lists of
                matplotlib images; a subject from different angles. One key
                (angles) corresponds to the respective angles.
                keys: 'mask', 'curv', 'parc', 'xcoord', 'ycoord', 'zcoord'
                    'angles', and optionally other keys for other
                    stat_map channels.
                vals: lists of images of the corresponding modality. Each
                    list entry should be a different view of the same subject.
            ns: (int, optional) new pixel size of downsampled square image.
            mode: (string, optional - defaults to 'HBN') or else 'OTS', or else
                'OTS-MFS-GAP', specifying which image input is to be processed
                via process_parc_img.
            resample: (optional) PIL method to use for image downsampling
        Returns:
            npy_dict: dictionary with the same keys, each a list of the processed
                input plt figures. The figures are processed by downsampling,
                conversion to consistent channels, and conversion to npy arrays.
    """
    PROCESS_FUNCTIONS = {'mask': process_mask_img,
                         'curv': process_curv_img,
                         'parc': lambda img:
                         process_parc_img(img, resample=resample),
                         'xcoord': process_coord_img,
                         'ycoord': process_coord_img,
                         'zcoord': process_coord_img
                         }
    if mode == 'OTS':
        PROCESS_FUNCTIONS['parc'] = lambda img: \
            process_parc_img(img, pal=PAL_OTS, resample=resample)
    elif mode == 'OTS-MFS-GAP':
        PROCESS_FUNCTIONS['parc'] = lambda img: \
            process_parc_img(img, pal=PAL_OTS_MFS_GAP, resample=resample)
    DEFAULT_CHANNEL_FUNCTION = process_curv_img
    npy_dict = {}
    npy_dict['angles'] = img_dict['angles']
    nangles = len(img_dict['angles'])

    for key, val in img_dict.items():
        if key == 'angles':  # skip angles; no figures stored in this key
            continue
        if key in PROCESS_FUNCTIONS:
            fn = PROCESS_FUNCTIONS[key]
        else:
            fn = DEFAULT_CHANNEL_FUNCTION
        npy_dict[key] = [fn(fig_to_PIL(fig)) for fig in img_dict[key]]

    return npy_dict


def px2v_from_np_dict(np_dict, mesh_coords):
    """ Wrapper of make_pixel_map to work with
        data object np_dict.
        Like make_pixel_map, this function calculates
        a dictionary of image pixels (i, j) to lists of vertices [v1, ...]
        in a cortical mesh, such that each vertex is listed only in the value
        of its closest pixel (i, j) key.
        Args:
            np_dict: a dictionary with keys : values given by
                'xcoord': list of np array versions of images of x-coordinate
                    of a cortical mesh.
                'ycoord': same as above, but for the y-coordinate.
                'zcoord': same as above, but for the z-coordinate.
        Returns:
            a copy of np_dict with additional keys : values
                'px_coord': a list of arrays (n_px, n_px, 3) each containing
                    the vertex (x, y, z)-coordinate mapped to the (i, j) pixel
                'px2v_dict': a list of dictionaries with keys (i, j) pixels
                    and values equal to lists of vertices closer to (i, j) than
                    to any other pixel (k, l).

        For further documentation see 'make_pixel_map'.
    """
    import copy
    assert 'angles' in np_dict.keys()

    angles = np_dict['angles']
    px_coord_list = []
    px2v_dict_list = []

    for i, a in enumerate(angles):
        x = np_dict['xcoord'][i]
        y = np_dict['ycoord'][i]
        z = np_dict['zcoord'][i]
        px_coords, px2v_dict = make_pixel_map(x, y, z, mesh_coords, a)
        px_coord_list.append(px_coords)
        px2v_dict_list.append(px2v_dict)

    np_px_dict = copy.deepcopy(np_dict)
    np_px_dict['px_coord'] = px_coord_list
    np_px_dict['px2v_dict'] = px2v_dict_list
    return np_px_dict


def save_subject_npys(sub, np_dict, save_xdir, save_ydir,
                      extra_channel_keys=None,
                      save_px2v_dir=None,
                      save_pxcoord_dir=None):
    """ Given subject images in the form of np_dict,
        saves .npy files corresponding to UNet x and y
        into two respective directories.
        Also stitches mask channels into single x array.
    """
    import pickle

    angles = np_dict['angles']
    if extra_channel_keys is None:
        extra_channel_keys = []

    for i, a in enumerate(angles):

        np_curv = np_dict['curv'][i]
        np_mask = np_dict['mask'][i]
        nps = [np_curv]
        for k in extra_channel_keys:
            nps.append(np_dict[k][i])
        nps.append(np_mask)
        onehot_indices = [1 + len(extra_channel_keys)]

        # create single np array
        np_x = stitch_nps(nps, onehot_indices=onehot_indices, n=None)
        np_y = np_dict['parc'][i]
        base_title = f'{sub}-{a[0]:.2f}-{a[1]:.2f}'

        x_title = f'{base_title}-curv.npy'
        y_title = f'{base_title}-parc.npy'

        fp_x = os.path.join(save_xdir, x_title)
        fp_y = os.path.join(save_ydir, y_title)

        with open(fp_x, 'wb') as f:
            np.save(f, np_x)

        with open(fp_y, 'wb') as f:
            np.save(f, np_y)

        # optionally, save px2v_dict
        if save_px2v_dir is not None:
            px2v = np_dict['px2v_dict'][i]
            px2v_title = f'{base_title}-px2v.pkl'
            fp_px2v = os.path.join(save_px2v_dir, px2v_title)

            with open(fp_px2v, 'wb') as f:
                pickle.dump(px2v, f)

        # optionally, save pxcoord
        if save_pxcoord_dir is not None:
            pxcoord = np_dict['px_coord'][i]
            pxcoord_title = f'{base_title}-pxcoord.npy'
            fp_pxcoord = os.path.join(save_pxcoord_dir, pxcoord_title)

            with open(fp_pxcoord, 'wb') as f:
                np.save(f, pxcoord)

    return


