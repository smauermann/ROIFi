import os
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import gamma
import matplotlib.pyplot as plt

HOME = os.path.expanduser("~")
DESKTOP = os.path.join(HOME, 'Desktop')
ROOT = 'Google_Drive/Master_Thesis/ROI_project'
RESULT_DIR = os.path.join(HOME, ROOT, 'results')
sub_dir = 'spectral_highres_corr_unnormalized'
result_sub_dir = os.path.join(RESULT_DIR, sub_dir)

data = np.load(os.path.join(result_sub_dir, 'spectral_highres_corr_unnormalized.npy'))
x, y = data.T

f = interp1d(x, y)
x_new = np.arange(np.min(x), np.max(x) + 1)
y_new = f(x_new)
np.save('/Users/Stephan/Desktop/spectral_highres_corr_unnormalized_interpol.npy', y_new, allow_pickle=True, fix_imports=True)
# y_mean = np.mean(y[:20])
# y_new -= y_mean

#print(fit_alpha, fit_loc, fit_beta)

#ax.plot(x_new, gamma.pdf(x_new, fit_alpha, fit_loc, fit_beta), 'r-', lw=5, alpha=0.6, label='gamma pdf')
