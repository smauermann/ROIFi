from nilearn.image import resample_img
import numpy as np
import nibabel as nib
import os
try:
    import cPickle as pickle
except:
    import pickle
import gzip

HOME = os.path.expanduser('~')
ROOT = os.path.join(HOME, 'Google_Drive/Master_Thesis/ROI_project')
FOLDER = 'atlas'
IMG = 'jhu189_grey_only.nii'
img_path = os.path.join(ROOT, FOLDER, IMG)
# use interpolation='nearest' for masks and interpolation='continuous'
# for normal images
resampling = 0
resampling_str = 'r%s' % resampling if resampling != 0 else ''
img_resampled = resample_img(img_path, interpolation='nearest',
                             target_affine=np.diag(resampling * np.ones(3)),
                             target_shape=None)

nib.save(img_resampled, os.path.join(ROOT, FOLDER, resampling_str + IMG))
