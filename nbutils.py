## valid for HBN

import os
import numpy as np
import pandas as pd

def parse(st, j):
    dcs = st[:-4].split('-')
    for i in range(len(dcs)):
        if dcs[i] == '':
            dcs[i+1] = '-' + dcs[i+1]
    dcs = [d for d in dcs if d != '']
    return dcs[j]

def parse_name(st):
    return 'sub-' + parse(st, 1)

def parse_angle1(st):
    return float(parse(st, 2))

def parse_angle2(st):
    return float(parse(st, 3))

def parse_trans1(st):
    return int(parse(st, 5))

def parse_trans2(st):
    return int(parse(st, 6))


extract_dict = {'EID': parse_name, 'Angle1': parse_angle1,
                'Angle2': parse_angle2
                }

extract_dict_trans = {'EID': parse_name, 'Angle1': parse_angle1,
                      'Angle2': parse_angle2, 'Shift1': parse_trans1,
                      'Shift2': parse_trans2
                      }

extract_dict_trans_mask = {'EID': parse_name, 'Angle1': parse_angle1,
                           'Angle2': parse_angle2,
                           'Shift1': (lambda x: int(parse(x, 6))),
                           'Shift2': (lambda x: int(parse(x,7)))
                           }


def make_fn_df(fn_list: list, extract_dict: dict,
               fn_colname: str = 'Filename') -> pd.DataFrame:
    
    """ Parses each filename in a list into different fields and returns them
        as a single dataframe."""
    
    df = pd.DataFrame({fn_colname: fn_list})
    for colname, extract_fn in extract_dict.items():
        df[colname] = df[fn_colname].apply(lambda x: extract_fn(x))
    return df


def get_subjects_df(directory, extract_dict=None):
    
    """ Given a directory with subject-related files,
        this function parses filenames to extract data of
        the angle and shift parameters for each subject
        and to package them into a single dataframe.
        Args:
            directory (string): where to find subject files
            extract_dict (dict): which values to extract from
                each subject file, with keys the column names in
                the returned dataframe, and values the functions
                to parse out the corresponding value.
                If None (default), this extracts only EID and Angle1, Angle2.
        Returns:
            dataframe (pandas.DataFrame) with columns of parsed filename,
                according to keys of extract_dict.
    """

    import os
    import pandas as pd
    
    fns = os.listdir(directory)
    
    if extract_dict is None:
        extract_dict = {'EID': parse_name, 'Angle1': parse_angle1,
            'Angle2': parse_angle2}
    
    return make_fn_df(fns, extract_dict)

LABELS_TO_PLOT = [2, 19, 21, 23, 24, 25, 30, 37, 38, 50, 51, 57, 58, 59, 60, 61, 63, 65]
hbn_dir = '/oak/stanford/groups/jyeatman/HBN/BIDS_curated/derivatives/freesurfer'


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
        assert nonrandom.shape[1] == 2
    else:
        n = ntotal
    random_angles = make_random_angles(n, center=center, scale=scale)

    angles = None
    if nonrandom_angles is not None:
        angles = np.concatenate((nonrandom, random_angles))
    else:
        angles = random_angles
    return angles


def get_freesurfer_subject(sub_dir, mesh_files, data_files,
                           surf_dir='surf'):
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
    from nilearn import surface
    import os
    
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


def get_freesurfer_subject_with_parc(sub_dir, mesh_files, data_files,
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
        label_files = ['lh.aparc.a2009s.annot']
    surf_dict = get_freesurfer_subject(sub_dir, mesh_files, data_files)
    label_dict = get_freesurfer_subject(sub_dir, mesh_files=None,
                                        data_files=label_files, surf_dir='label')
    surf_dict.update(label_dict)
    return surf_dict


def make_subject_images(sub, mesh, angles=None, nonrandom_angles=None):
    parc_path = f'{hbn_dir}/{sub}{label_subdir}/{parc_filename}'
    curv_path = f'{hbn_dir}/{sub}{surf_subdir}/{curv_filename}'
    infl_path = f'{hbn_dir}/{sub}{surf_subdir}/{infl_filename}'

    # check files exist
    if not (os.path.exists(parc_path)
            and os.path.exists(curv_path)
            and os.path.exists(infl_path)
            ):
        return

    selected_parc = np.array([LABELS_TO_PLOT.index(l) if l in labels_to_plot else -1 for l in parc])

    if angles is None:
        angles = make_angles(20, nonrandom_angles=nonrandom_angles)
    for a in angles:
        fig, ax = plt.subplots(figsize=(8, 8))
        plotting.plot_surf_roi(infl, selected_parc
                               , view=(a[0], a[1])
                               # ,bg_map=test_curv
                               # ,bg_on_data=True
                               , figure=fig
                               , cmap='tab20'
                               , output_file=f'{out_parc}/{sub}-{a[0]}-{a[1]}-parc.png'
                               # ,threshold=25.0
                               # colorbar=True
                               )

        fig, ax = plt.subplots(figsize=(8, 8))
        plotting.plot_surf_roi(infl, selected_parc
                               , view=(a[0], a[1])
                               , bg_map=curv
                               # ,bg_on_data=True
                               , figure=fig
                               , cmap='tab20'
                               , threshold=25.0
                               , output_file=f'{out_curv}/{sub}-{a[0]}-{a[1]}-curv.png'
                               # colorbar=True
                               )
    return

