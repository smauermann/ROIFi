import os

from rf_all import PearsonMerger, SpectralClustering

HOME = os.path.expanduser("~")
PROJECT_DIR = 'Google_Drive/Master_Thesis/ROI_project'
DATA_DIR = os.path.join(HOME, PROJECT_DIR, 'data')
RESULT_DIR = os.path.join(HOME, PROJECT_DIR, 'results')
MASK_IMG = os.path.join(HOME, PROJECT_DIR, 'r5mask.nii')

rf = PearsonMerger(random_seed=42)#, volumes=DATA_DIR, mask_img=MASK_IMG)
rf.run(r_threshold=0.5, p_threshold=0.05, min_cluster_size=None)
rf.draw_clusters(cubes=False)
