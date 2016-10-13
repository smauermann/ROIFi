import numpy as np
import nibabel as nib
from rf_new import PearsonMerger
import os

HOME = os.path.expanduser("~")
DESKTOP = os.path.join(HOME, 'Desktop')
ROOT = 'Google_Drive/Master_Thesis/ROI_project'
RESULT_DIR = os.path.join(HOME, ROOT, 'results')

pm = PearsonMerger(volumes=None, mask_img=None,
                   normalize_volumes=True, verbose=1)
clusters = os.path.join(RESULT_DIR, 'images_pearson_all_corr_norm/pearson_rt0.88_pt0.05_300916_1648/cluster_img.nii.gz')
correlation_map = pm.mean_cluster_correlations(clusters)
print(correlation_map)
np.save(os.path.join(DESKTOP, 'correlation_map.npy'), correlation_map)
