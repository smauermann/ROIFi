import os

from roi_finder_class import PearsonMerger, SpectralClustering

HOME = os.path.expanduser("~")
PROJECT_DIR = 'Google_Drive/Master_Thesis/ROI_project'
DATA_DIR = os.path.join(HOME, PROJECT_DIR, 'data')
RESULT_DIR = os.path.join(HOME, PROJECT_DIR, 'results')
MASK_IMG = os.path.join(HOME, PROJECT_DIR, 'r5mask.nii')

RF = SpectralClustering(volumes=DATA_DIR, mask_img=MASK_IMG)
RF.find_clusters()
