import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle

from util import load_vmaps
from pipeline_utilities import fig_to_PIL

from nilearn import plotting

start = int(sys.argv[1])
end = int(sys.argv[2])

hbn_dir = '/oak/stanford/groups/jyeatman/HBN/BIDS_curated/derivatives/freesurfer'
hbn_surf_dir = 'surf'
hbn_parc_dir = 'label'
anat_load_dir = hbn_dir
vmap_load_dir = '/scratch/groups/jyeatman/samjohns-projects/notebooks/ccnn/results/hbn_ots_vmaps_2022-11-27'

image_dir = '/scratch/groups/jyeatman/samjohns-projects/notebooks/ccnn/results/swipes_stack'
# os.makedirs(image_dir, exist_ok=True)
# eids = list(set([e.split('_')[-1][:-4] for e in os.listdir(vmap_load_dir)]))

eids = pickle.load(open('/scratch/groups/jyeatman/samjohns-projects/notebooks/ccnn/results/eids_to_do.pkl', 'rb'))
eids.sort()  # guarantees eids are in the same order across parallel jobs

for eid in eids[start:end]:
    sub_data = load_vmaps(eid, anat_load_dir, vmap_load_dir)

    mesh = sub_data['mesh']
    curv = sub_data['curv']
    vmap = sub_data['vmap']

    binc = -1.0 * np.ones_like(curv)
    binc[curv >= 0.0] = 1.0

    prob = vmap.mean(axis=0)[:, 2]
    ots = 0.0 * np.ones_like(prob)
    ots[prob > float(3/7)] = 3.0

    fig1, _ = plt.subplots(figsize=(8, 8))

    plotting.plot_surf_stat_map(
        mesh,
        binc,
        view=(270., 90.),
        figure=fig1,
        vmax=2.0,
        cmap='gray',
        colorbar=False
    )

    fig2, _ = plt.subplots(figsize=(8, 8))

    plotting.plot_surf_stat_map(
        mesh,
        ots,
        bg_map=binc,
        view=(270., 90.),
        threshold=2.0,
        figure=fig2,
        vmax=5.0,
        cmap='PiYG',
        colorbar=False
    )

    bg_np = np.array(fig_to_PIL(fig1))
    ots_np = np.array(fig_to_PIL(fig2))

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(bg_np)
    ax.set_axis_off()
    plt.savefig(f'{image_dir}/x_{eid}.png')

    ax.imshow(ots_np, alpha=0.5)
    ax.set_axis_off()
    plt.savefig(f'{image_dir}/y_{eid}.png')
    plt.close('all')
