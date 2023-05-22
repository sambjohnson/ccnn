import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pipeline_utilities as pu
from nilearn import surface

start = int(sys.argv[1])
end = int(sys.argv[2])
# start, end = 0, 1  # change values during testing
data_dir = '/oak/stanford/groups/jyeatman/LMB_Stanford/LMB_BIDS/derivatives/freesurfer'
label_subdir = 'label'
surf_subdir = 'surf'

subjects = [s for s in os.listdir(data_dir) if 'sub' in s]
jewelia_labeled_subjects = ['sub-107', 'sub-108', 'sub-1210', 'sub-1211', 'sub-1215', 
                            'sub-1234', 'sub-1335', 'sub-1339', 'sub-1380', 'sub-1395', 
                            'sub-1444', 'sub-1452', 'sub-1453', 'sub-1468', 'sub-173', 
                            'sub-230', 'sub-309', 'sub-984']
subjects = [s for s in subjects if s in jewelia_labeled_subjects]
subjects.sort()

sub_dfs = []

for sub in subjects:

    curv_file = os.path.join(data_dir, sub, surf_subdir, 'lh.curv')
    sulc_file = os.path.join(data_dir, sub, surf_subdir, 'lh.sulc')
    infl_file = os.path.join(data_dir, sub, surf_subdir, 'lh.inflated')
    label_dir = os.path.join(data_dir, sub, label_subdir)
    label_files = [f for f in os.listdir(label_dir) if ('.label' in f) and ('lh.' in f)]

    surf = surface.load_surf_mesh(infl_file)
    curv = surface.load_surf_data(curv_file)
    sulc = surface.load_surf_data(sulc_file)

    labels = []
    for l in label_files:
        labels.append(surface.load_surf_data(os.path.join(label_dir, l)))

    sub_df = pd.DataFrame(
        {
            'Subject': sub,
            'Filename': label_files,
            'Label': labels,
            'Mesh': [surf] * len(labels),
            'Curv': [curv] * len(labels),
            'Sulc': [sulc] * len(labels)
        }
    )
    sub_dfs.append(sub_df)

df = pd.concat(sub_dfs)

all_label_files = list(df['Filename'].unique())
all_labels = sorted([l.split('.')[1] for l in all_label_files if 'lh.' in l])

l_inc = ['OTS', 
         'mfs', 
         'ptCoS', 
         'CoS', 
         'atCoS']

df['LabelName'] = df['Filename'].apply(lambda l: l.split('.')[1])
df_filtered = df[df['LabelName'].isin(l_inc)]
label_to_index = dict(zip(l_inc, list(range(2, len(l_inc) + 2))))
df_filtered['LabelIndex'] = df_filtered['LabelName'].apply(lambda name: label_to_index[name])

def make_subject_stat_map(sub, df, return_mesh=True):
    sub_df = df[df['Subject'] == sub]

    sub_mesh = sub_df['Mesh'].iloc[0]
    sub_labs = list(sub_df['Label'])
    sub_inds = list(sub_df['LabelIndex'])

    c = sub_mesh.coordinates.shape[0]
    m = np.ones(c)

    for l, i in zip(sub_labs, sub_inds):
        m[l] = float(i)

    if return_mesh:
        return m, sub_mesh
    else:
        return m


def get_subject_labels(sub, df):
    curv = df[df['Subject'] == sub]['Curv'].iloc[0]
    stat, mesh = make_subject_stat_map(sub, df)
    return mesh, stat, curv


ots_dir = data_dir
save_base_dir = '/scratch/groups/jyeatman/samjohns-projects/' \
                'data/atlas/lmb-ots'

save_x_subdir = 'xs'
save_y_subdir = 'ys'
save_xdir = os.path.join(save_base_dir, save_x_subdir)
save_ydir = os.path.join(save_base_dir, save_y_subdir)

save_px2v_subdir = 'px2v'
save_pxcoord_subdir = 'pxcoord'
save_px2v_dir = os.path.join(save_base_dir, save_px2v_subdir)
save_pxcoord_dir = os.path.join(save_base_dir, save_pxcoord_subdir)

os.makedirs(save_base_dir, exist_ok=True)
os.makedirs(save_xdir, exist_ok=True)
os.makedirs(save_ydir, exist_ok=True)
os.makedirs(save_px2v_dir, exist_ok=True)
os.makedirs(save_pxcoord_dir, exist_ok=True)

ots_subjects = list(df_filtered['Subject'].unique())

# main loop
for sub in ots_subjects[start:end]:

    mesh, stat, curv = get_subject_labels(sub, df_filtered)
    sulc = df_filtered[df_filtered['Subject'] == sub]['Sulc'].iloc[0]
    extra_channels_dict = {'sulc': sulc}

    # pipeline (below):
    # 1. create plt figures
    # 2. process (downsample, grayscale, extract channels) -> np array
    # 3. get px2v data from coordinate images

    nangles_inner = 5
    nangles_total = 40
    nangle_iterations = nangles_total // nangles_inner

    for i in range(nangle_iterations):
        fig_dict = pu.make_subject_images(
            mesh,
            curv,
            stat,
            extra_channels_dict=extra_channels_dict,
            nangles=nangles_inner,
            mode='OTS-MFS-GAP'
        )
        np_dict = pu.process_figs(fig_dict, mode='OTS-MFS-GAP')
        np_px_dict = pu.px2v_from_np_dict(
            np_dict,
            mesh_coords=mesh.coordinates
        )
        plt.close('all')  # clear all matplotlib plots
        pu.save_subject_npys(
            sub,
            np_px_dict,
            save_xdir,
            save_ydir,
            extra_channel_keys=['sulc'],
            save_px2v_dir=save_px2v_dir,
            save_pxcoord_dir=save_pxcoord_dir
        )
        del np_dict
        del np_px_dict