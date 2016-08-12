import os

from rf_new import PearsonMerger, SpectralClustering
from utils.load_data import load_structural

HOME = os.path.expanduser("~")
PROJECT_DIR = 'Google_Drive/Master_Thesis/ROI_project'
RESULT_DIR = os.path.join(HOME, PROJECT_DIR, 'results')
MASK_IMG = os.path.join(HOME, PROJECT_DIR, 'r5mask.nii')

METHODS = ['PearsonMerger', 'SpectralClustering']
method = METHODS[0]
img_files = load_structural(project=2, smoothing=8, resampling=5, type=1)[0]

if method == 'PearsonMerger':
    rf = PearsonMerger(volumes=img_files, mask_img=MASK_IMG,
                       normalize_volumes=True, verbose=2)
    rf.run(output_path=RESULT_DIR, r_threshold=None, p_threshold=None,
           min_cluster_size=10, draw=None, auto_threshold=True)
elif method == 'SpectralClustering':
    rf = SpectralClustering(volumes=img_files, mask_img=MASK_IMG,
                            normalize_volumes=True, verbose=0)
    rf.run(output_path=RESULT_DIR, n_clusters=50, draw=None)
