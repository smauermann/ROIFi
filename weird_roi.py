import os
import numpy as np
import nibabel as nib
from rf_new import SpectralClustering, PearsonMerger, NoClustersError
import re
import matplotlib.pyplot as plt
from utils.load_data import load_structural
from roi_classifier import roi_classify
import seaborn as sns
import json
from figures import plot_figure
import socket

HOME = os.path.expanduser("~")
DESKTOP = os.path.join(HOME, 'Desktop')
ROOT = 'Google_Drive/Master_Thesis/ROI_project'
RESULT_DIR = os.path.join(HOME, ROOT, 'results')
MASKS = {0: 'alc_P2_mask.nii', 3: 'r3alc_P2_mask.nii', 5: 'r5alc_P2_mask.nii'}
TYPEDICT = {'all': None, 'patients': 1, 'controls': 0}

find_rois = True
classify = False
old_classify = False

METHOD = ['PearsonMerger', 'SpectralClustering'][1]
ESTIMATOR = ['weird', 'svc'][0]
SCORING = 'recall'
labels_scores = []

RESAMPLING = 5
AGE_CORR = True
NORMALIZED = True
TYPE = 'all'
DRAW = 'brain_map'

normalized_str = 'norm' if NORMALIZED else 'unnorm'
method_str = 'pearson' if METHOD == 'PearsonMerger' else 'spectral'
corr_str = 'corr' if AGE_CORR else 'uncorr'
sub_dir = 'images_%s_%s_%s_%s' % (method_str, TYPE, corr_str, normalized_str)
result_sub_dir = os.path.join(RESULT_DIR, sub_dir)
# check if sub_dir exists otherwise create it:
if not os.path.isdir(result_sub_dir) and not os.path.isfile(result_sub_dir):
    os.makedirs(result_sub_dir)
    print('Subdir %s created!' % result_sub_dir)

# type: patients=1, controls=0
# use age-corrected data and RESAMPLING 5 as standard, gave best results so far
img_files, labels, subjects = load_structural(project=2, smoothing=8,
                                              resampling=RESAMPLING,
                                              type=TYPEDICT[TYPE],
                                              corrected=AGE_CORR)

host = socket.gethostname()
if host == 'smauermann-2513k.charite.de' or host == 'Neurotronix-Macbook.local':
    mask = os.path.join(HOME, ROOT, MASKS[RESAMPLING])
else:
    mask = os.path.join(HOME, 'Dropbox', 'Stephan', MASKS[RESAMPLING])


if find_rois:
    if METHOD == 'PearsonMerger':
        rf = PearsonMerger(volumes=img_files, mask_img=mask,
                           normalize_volumes=NORMALIZED, verbose=1)
        r_range = np.linspace(0.8, 0.96, 5)
        for r in r_range:
            try:
                rf.run(output_path=result_sub_dir, r_threshold=r,
                       p_threshold=0.05, min_cluster_size=10, draw=DRAW,
                       auto_threshold=False, atlas=None, cluster_means=classify)
            except NoClustersError as e:
                print(e)
                break
            if classify:
                ls = rf.do_classification(estimator=ESTIMATOR, labels=labels)
                labels_scores.append(ls)
    elif METHOD == 'SpectralClustering':
        rf = SpectralClustering(volumes=img_files, mask_img=mask,
                                normalize_volumes=NORMALIZED, verbose=1)
        cluster_range = [10,100,250,500,1000]
        for x in cluster_range:
            rf.run(output_path=result_sub_dir, n_clusters=x, draw=DRAW,
                   distance_measure='correlation', atlas=None,
                   cluster_means=classify)
            if classify:
                ls = rf.do_classification(estimator=ESTIMATOR, labels=labels)
                labels_scores.append(ls)

if classify:
    labels_scores = np.array(labels_scores)
    np.save(result_sub_dir, labels_scores)

# result_folder = os.path.join(HOME, ROOT, RESULT_DIR, 'spectral_1500_correlation_unnormalized')
result_folder = result_sub_dir

if old_classify:
    if METHOD == 'SpectralClustering':
        result_dirs = [x for x in os.listdir(result_folder) if x.startswith('spectral_')]
        parameter_labels = [re.search('(?<=spectral_)[0-9]+(?=clusters)', x) for x in result_dirs]
        parameter_labels = [int(x.group()) for x in parameter_labels]
    elif METHOD == 'PearsonMerger':
        result_dirs = [x for x in os.listdir(result_folder) if x.startswith('pearson_')]
        parameter_labels = [re.search('(?<=pearson_rt)[0-1].[0-9]+', x) for x in result_dirs]
        parameter_labels = [float(x.group()) for x in parameter_labels]
    data_paths = [os.path.join(result_folder, x, 'cluster_means.npy') for x in result_dirs]
    all_data = [np.load(x) for x in data_paths]
    n_subj = len(subjects)
    # dumped classificaton scores
    # dumped_scores_file = os.path.join(result_sub_dir, '%s_%s_scores.npy' % (ESTIMATOR, SCORING))
    dumped_scores_file = os.path.join(result_sub_dir, 'weird_recall_scores.npy')
    # do classification if no dumped data exists
    if not os.path.isfile(dumped_scores_file):
        print('%s classification running ...' % ESTIMATOR)
        results = []
        for l, d in zip(parameter_labels, all_data):
            score = roi_classify(d, labels, n_subj, estimator=ESTIMATOR,
                                 scoring=SCORING)
            results.append((l, score))

        results = np.array(results)
        # dump scores for later
        np.save(dumped_scores_file, results)
    # load classification data from pickle
    else:
        print('Loading dumped classification scores!')
        results = np.load(dumped_scores_file)

    if METHOD == 'PearsonMerger':
        # load metadata dicts for mean_cluster_sizes and all_n_clusters
        metadata_dicts = []
        for d in result_dirs:
            dict_file = os.path.join(result_folder, d, 'metadata.json')
            with open(dict_file) as df:
                data = json.load(df)
                metadata_dicts.append(data)
        mean_cluster_sizes = [x['mean_cluster_size'] for x in metadata_dicts]
        all_n_clusters = [x['cluster_count'] for x in metadata_dicts]
        #normalized = metadata_dicts[-1]['normalized']
    else:
        mean_cluster_sizes = None
        all_n_clusters = None

    scores = results[:, 1]
    parameter_labels = results[:, 0]

    fig = plot_figure(scores, parameter_labels, method=METHOD, age_corr=AGE_CORR,
                      normalized=NORMALIZED, mean_cluster_sizes=None,
                      all_n_clusters=None, regression=None)
    fig.savefig(os.path.join(result_sub_dir, 'balanced_acc_%s.png' % sub_dir))
