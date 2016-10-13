import os
import pickle
import warnings

import nibabel as nib
import numpy as np
import pandas as pd
import socket

HOME = os.path.expanduser("~")
if socket.gethostname() == 'malin':
    ROOT_STRUCT = '/local/matthiasg/mpr/'
    PATH_LABELS = os.path.join(HOME, 'Dropbox/Stephan/labels2.pkl')
else:
    ROOT_STRUCT = os.path.join(HOME, 'Google_Drive/Master_Thesis/ROI_project/data')
    PATH_LABELS = os.path.join(ROOT_STRUCT, 'labels2.pkl')

TARGET_DTYPE_DICT = {'label': np.int8, 'relapse': np.int8, 'type': np.int8,
                     'center': np.int8, 'ADS': np.int8, 'age': np.float16,
                     'Drinking_timespansince_firstDrunken': np.float16,
                     'Drinking_kg_lifetime': np.float16,
                     'Drinking_timespansince_OnseBingeing': np.float16,
                     'Drinking_Frequency_Bingeing_lifetime': np.float16,
                     'Drinking_kg_perdrinkyear_v10': np.float16}

ROOT_VALIDATION = '/data/fmri/simu/'
PATH_SPSS = os.path.join(HOME, 'science/data/KAP BL v20.sav')


MAPPING = dict(
    age='Age_baseline',
    sex='Sex',
    smoking='FTND_I_Sum',
    center='BD'
)


# project=2
def load_structural_stephan(target='label', project=(1, 2), type=None, modality='grey',
                    smoothing=None, modulation='m0', resampling=None,
                    exclude=tuple(), load_images=False, alternative_root=None, corrected=False):
    metadata = pickle.load(open(PATH_LABELS, 'rb'))
    if alternative_root is None:
        root = ROOT_STRUCT
    else:
        root = alternative_root
    subjects = np.unique([int(s.split('-')[0][-4:]) for s in os.listdir(root) if s.endswith('.nii')])

    smoothing_str = '' if smoothing is None else 's%g' % smoothing
    resampling_str = '' if resampling is None else 'r%g' % resampling
    cor_str = '_cor' if corrected else ''

    modality_dict = dict(grey=1, white=2, csf=3)

    labels_all = np.array([metadata[target][s] if (s in metadata[target].keys() and metadata[target][s] is not None and metadata['befundung'][s] is not 2 and s not in exclude) else -1 for s in subjects],
                          dtype=TARGET_DTYPE_DICT[target])
    projects_all = np.array([metadata['project'][s] if (s in metadata['project'].keys() and metadata['project'][s] is not None and metadata['befundung'][s] is not 2) else -1 for s in subjects],
                            dtype=np.int8)
    images_all = [os.path.join(root, '%s%s%swp%gs%s-mpr%s.nii' %
                               (smoothing_str, resampling_str, modulation, modality_dict[modality], s, cor_str)) for s in subjects]

    subjects = [s for i, s in enumerate(subjects) if (labels_all[i] != -1) & np.in1d(projects_all[i], project)]
    labels = labels_all[(labels_all != -1) & np.in1d(projects_all, project)]
    images = [images_all[i] for i, (l, p) in enumerate(zip(labels_all, projects_all)) if (l != -1) & np.in1d(p, project)]

    if type is not None:
        labels = np.array([labels[i] for i, s in enumerate(subjects) if metadata['label'][s] == type])
        images = [images[i] for i, s in enumerate(subjects) if metadata['label'][s] == type]
        subjects = [s for s in subjects if metadata['label'][s] == type]

    if load_images:
        images = [nib.load(img) for img in images]

    if target == 'label':
        print('Patients (1): %s# Controls(0): #%s' % (sum(labels == 1), sum(labels == 0)))
    return images, labels, subjects


def load_structural_malin(target='label', project=(1, 2), type=None, modality='grey',
                    smoothing=None, modulation='m0', resampling=None,
                    exclude=tuple(), load_images=False, alternative_root=None, corrected=False):
    metadata = pickle.load(open(PATH_LABELS, 'rb'))
    if alternative_root is None:
        root = ROOT_STRUCT
    else:
        root = alternative_root
    # subjects = np.unique([int(s.split('-')[0][-4:]) for s in os.listdir(root) if s.endswith('.nii')])
    subjects = [int(s) for s in list_dirs(root, exceptions='mpr')]

    smoothing_str = '' if smoothing is None else 's%g' % smoothing
    resampling_str = '' if resampling is None else 'r%g' % resampling
    cor_str = '_cor' if corrected else ''

    modality_dict = dict(grey=1, white=2, csf=3)

    # labels_all = np.array([metadata[target][s] if (s in metadata[target].keys() and metadata[target][s] is not None and metadata['befundung'][s] is not 2 and os.path.exists(os.path.join(root, str(s), 'mpr', '%s%swp%gs%s-mpr.nii' % (smoothing_str, modulated_str, modality_dict[modality], s))) and s not in exclude) else -1 for s in subjects],
    #                       dtype=TARGET_DTYPE_DICT[target])
    labels_all = np.array([metadata[target][s] if (s in metadata[target].keys() and metadata[target][s] is not None and metadata['befundung'][s] is not 2 and s not in exclude) else -1 for s in subjects],
                          dtype=TARGET_DTYPE_DICT[target])
    projects_all = np.array([metadata['project'][s] if (s in metadata['project'].keys() and metadata['project'][s] is not None and metadata['befundung'][s] is not 2) else -1 for s in subjects],
                            dtype=np.int8)
    images_all = [os.path.join(root, str(s), 'mpr', '%s%s%swp%gs%s-mpr%s.nii' %
                               (smoothing_str, resampling_str, modulation, modality_dict[modality], s, cor_str)) for s in subjects]

    subjects = [s for i, s in enumerate(subjects) if (labels_all[i] != -1) & np.in1d(projects_all[i], project)]
    labels = labels_all[(labels_all != -1) & np.in1d(projects_all, project)]
    images = [images_all[i] for i, (l, p) in enumerate(zip(labels_all, projects_all)) if (l != -1) & np.in1d(p, project)]

    if type is not None:
        labels = np.array([labels[i] for i, s in enumerate(subjects) if metadata['label'][s] == type])
        images = [images[i] for i, s in enumerate(subjects) if metadata['label'][s] == type]
        subjects = [s for s in subjects if metadata['label'][s] == type]

    if load_images:
        images = [nib.load(img) for img in images]

    # if target == 'label':
    #     print('Patients (1): %s# Controls(0): #%s' % (sum(labels == 1), sum(labels == 0)))
    return images, labels, subjects

if socket.gethostname() == 'malin':
    load_structural = load_structural_malin
else:
    load_structural = load_structural_stephan


def list_dirs(root, fullpath=False, exceptions=None):
    if exceptions is None:
        exceptions = []
    elif isinstance(exceptions, str):
        exceptions = [exceptions]

    if fullpath:
        dirs = [os.path.join(root, name) for name in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, name))
                and name not in exceptions]
    else:
        dirs = [name for name in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, name))
                and name not in exceptions]

    return dirs

def load_fmri(exp='PIT', model='pit_1_1_1', cons=(1,), project=(1, 2), func_subj_id=lambda x: x, load_images=False):
    root = {'PIT': ROOT_PIT, 'TS': ROOT_TS}

    images = [[]] * len(cons)
    allsubjects, incomplete_subjects = [], [[]] * len(cons)
    for p in np.intersect1d([1, 2], project):
        path = os.path.join(root[exp], 'P%g' % p, '1stLevel_spm12', model)
        subjectpaths = list_dirs(path, fullpath=True)

        for s, spath in enumerate(subjectpaths):

            allsubjects += [int(func_subj_id(spath))] if func_subj_id(spath).isdigit() else func_subj_id(spath)

            for i, c in enumerate(cons):
                imagepath = os.path.join(spath, 'con_%04g.nii' % c)
                if os.path.exists(imagepath):
                    images[i] += [imagepath]
                else:
                    incomplete_subjects[i] += [allsubjects[-1]]
                    warnings.warn('Could not find file %s' % imagepath)

    complete_subjects = [s for s in allsubjects if s not in sum(incomplete_subjects, [])]

    return images[0] if len(images) == 1 else images, \
           complete_subjects, \
           allsubjects, \
           incomplete_subjects[0] if len(incomplete_subjects) == 1 else incomplete_subjects


def load_covariates(covnames, subjects):
    """
    Note: covariates have to be *lists* not numpy arrays for nipype
    :param covname: <string> name of covariate or List<string> names of covariates
    :param subjects: <list> list of subject id's
    :return:
    """

    if not isinstance(covnames, list):
        covnames = [covnames]

    metadata = pickle.load(open(PATH_LABELS, 'rb'))

    covariates = {cn: np.array([metadata[cn][s] if s in metadata[cn] else None for s in subjects], dtype=np.float) for cn in covnames}

    return covariates


def load_cov(covname, subjects):
    cov = load_covariates(covname, subjects)
    return cov[covname]


def load_validation_searchlight(mode):
    import json

    subjects = list_dirs(ROOT_VALIDATION)

    images = [os.path.join(ROOT_VALIDATION, s, 'roi', '%s_Fusiform_posterior_AAL.nii' % mode) for s in subjects]
    labels = [json.load(open(os.path.join(ROOT_VALIDATION, s, 'y_%s.json' % mode), 'r'))['y'] for s in subjects]

    return images, labels, subjects


def get_vars(varnames, subjects):

    vars = dict()
    for varname in varnames:
        vars[varname] = get_var(varname, subjects)

    return vars

def get_var(varname, subjects):

    if varname in MAPPING:
        varname = MAPPING[varname]

    try:
        values = DF.ix[subjects, varname].values
    except KeyError:
        print('[KeyError] Valid keys are:')
        print(DF.columns)
        raise

    if varname == 'Sex':
        values = [1 if k == 'male' else 2 for k in values]
    elif varname == 'BD':
        values = [1 if k == 'Berlin' else 2 for k in values]

    return np.array(values)
