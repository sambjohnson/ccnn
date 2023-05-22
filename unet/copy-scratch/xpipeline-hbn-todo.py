import numpy as np
import pandas as pd
import os
import sys

import pipeline_utilities as pu
import matplotlib.pyplot as plt

wd = '/scratch/groups/jyeatman/samjohns-projects/backups/samjohns-projects/samjohns-backups-2/data/atlas'
val = 'hbn_sulc_mapping_val.csv'
trn = 'hbn_sulc_mapping_trn.csv'

df_val = pd.read_csv(f'{wd}/{val}')
df_trn = pd.read_csv(f'{wd}/{trn}')

df = pd.concat([df_trn, df_val])
df.sort_values('EID', inplace=True)

subjects = list(df['EID'].unique())

start = int(sys.argv[1])
end = int(sys.argv[2])

hbn_dir = '/oak/stanford/groups/jyeatman/HBN/BIDS_curated/derivatives/freesurfer'
# hbn_subjects = [s for s in os.listdir(hbn_dir) if s[:4]=='sub-']

save_base_dir = '/scratch/groups/jyeatman/samjohns-projects/data/atlas-2'
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

parc_fn = 'lh.aparc.a2009s.annot'  # Destrieux parcellation
curv_fn = 'lh.curv'
mesh_fn = 'lh.inflated'

done_subjects = os.listdir(save_xdir)
done_eids = [s.split('-')[1] for s in done_subjects]

df.groupby('EID', as_index=False).count()

df_done = pd.DataFrame({'EID': done_eids, 'Dummy': 1})
df_counts = df_done.groupby('EID', as_index=False).count()
subjects_done = ['sub-' + s for s in list(df_counts[df_counts['Dummy'] == 70]['EID'])]
subjects_todo = list(set(subjects) - set(subjects_done))

print(os.getcwd())
csv = '/scratch/groups/jyeatman/samjohns-projects/backups/samjohns-projects/samjohns-backups-2/notebooks/ccnn-2/ccnn/unet/todo_subjects.csv'

subjects_todo_df_loaded = pd.read_csv(csv)
subjects_todo = list(subjects_todo_df_loaded['EID'])

for sub in subjects_todo[start:end]:
    
    sub_df = df[df['EID'] == sub]
    ang = list(sub_df[['Angle1', 'Angle2']].values)
    
    subj_fp = os.path.join(hbn_dir, sub)
    subject_data_exists = pu.freesurfer_subject_data_exists_parc(subj_fp, [mesh_fn],
                                                                 [curv_fn],
                                                                 label_files=[parc_fn])
    if not subject_data_exists:  # skip subject if required files don't exist
        continue
    subject_data = pu.get_freesurfer_subject_with_parc(subj_fp,
                                                       [mesh_fn],
                                                       [curv_fn],
                                                       label_files=[parc_fn])

    mesh = subject_data[mesh_fn]
    curv = subject_data[curv_fn]
    parc = subject_data[parc_fn]

    # pipeline (below):
    # 1. create plt figures
    # 2. process (downsample, grayscale, extract channels) -> np array
    # 3. get px2v data from coordinate images
    
    nangles_inner = 5
    nangles_total = len(ang)
    nangle_iterations = (nangles_total // nangles_inner) + 1

    for i in range(nangle_iterations):
        
        angle_batch = ang[(nangles_inner * i) : (nangles_inner * (i + 1))]
        if len(angle_batch) == 0:
            break
        
        fig_dict = pu.make_subject_images(mesh, curv, parc, angles=angle_batch) 
        np_dict = pu.process_figs(fig_dict)
        np_px_dict = pu.px2v_from_np_dict(np_dict, 
                                          mesh_coords=mesh.coordinates)
        plt.close('all')  # clear all matplotlib plots to save memory
        pu.save_subject_npys(sub, np_px_dict, save_xdir, save_ydir, 
                             save_px2v_dir=save_px2v_dir, 
                             save_pxcoord_dir=save_pxcoord_dir)
        del np_dict  # clear dictionaries to save memory
        del np_px_dict
        
    plt.close('all')
