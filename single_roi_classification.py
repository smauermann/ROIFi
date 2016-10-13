import os
import numpy as np
import nibabel as nib
from utils.load_data import load_structural
import numpy.ma as ma
from roi_classifier import roi_classify
import json

HOME = os.path.expanduser("~")
DESKTOP = os.path.join(HOME, 'Desktop')
ROOT = 'Google_Drive/Master_Thesis/ROI_project'
RESULT_DIR = os.path.join(HOME, ROOT, 'results')
TYPEDICT = {'all': None, 'patients': 1, 'controls': 0}

RESAMPLING = 5
TYPE = 'all'
AGE_CORR = True
NORMALIZED = True


atlas = nib.load(os.path.join(RESULT_DIR, 'images_pearson_all_corr_norm/pearson_rt0.88_pt0.05_300916_1648/cluster_img.nii.gz')).get_data()
metadata = json.load(open(os.path.join(RESULT_DIR, 'images_pearson_all_corr_norm/pearson_rt0.88_pt0.05_300916_1648/metadata.json')))
sizes = [v for k, v in sorted(metadata.items()) if k.startswith('size_cluster_')]
mask = nib.load(os.path.join(HOME, ROOT, 'r5alc_P2_mask.nii')).get_data()
mask_switched = mask ^ 1
img_files, labels, subjects = load_structural(project=2, smoothing=8,
                                              resampling=RESAMPLING,
                                              type=TYPEDICT[TYPE],
                                              corrected=AGE_CORR)
images = [nib.load(x).get_data() for x in img_files]
n_subjects = len(images)
# mask images and atlas
if mask is not None:
    for i, img in enumerate(images):
        images[i] = ma.array(img, mask=mask_switched, fill_value=0).filled()
    atlas = ma.array(atlas, mask=mask_switched, fill_value=0).filled()

# make cluster means on atlas ROIs
cluster_ids = list(np.unique(atlas[~np.isnan(atlas)]))
cluster_ids = cluster_ids[1:]
n_clusters = len(cluster_ids)
# n_subs x n_rois
cluster_means = np.empty((n_subjects, n_clusters))
for s, c in np.ndindex(cluster_means.shape):
    c_idx = np.where(atlas == cluster_ids[c])
    cluster_means[s, c] = np.mean(images[s][c_idx])

# iterate over cluster_means and classify
cluster_scores = np.empty((n_clusters, 2))
for i in range(n_clusters):
    data = cluster_means[:,i].reshape(-1,1)
    classification_score = roi_classify(data, labels, n_subjects,
                                         estimator='weird', verbose=0)
    cluster_scores[i, 0] = sizes[i]
    cluster_scores[i, 1] = classification_score
    print('cluster %s, recall score: %s' % (i, classification_score))

np.save(os.path.join(HOME, 'Desktop/pearsonmerger_scores.npy'), cluster_scores)
