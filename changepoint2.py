from scipy import optimize
from scipy.stats import norm, linregress
import numpy as np
import matplotlib.pylab as plt
from functools import partial
from scipy.interpolate import interp1d
import seaborn as sns
import os

HOME = os.path.expanduser("~")
DESKTOP = os.path.join(HOME, 'Desktop')
ROOT = 'Google_Drive/Master_Thesis/ROI_project'
RESULT_DIR = os.path.join(HOME, ROOT, 'results')

def changepoint2(data, xdata=None, cp1_range=None, cp2_range=None,
                 do_plot=False, verbose=True):

    def piecewise(x, m1, t1, m2, m3, cp1, cp2):
        t2 = x[int(cp1)] * (m1 - m2) + t1
        t3 = x[int(cp2)] * (m2 - m3) + t2
        y = np.hstack((m1 * x[:int(cp1)] + t1, m2 * x[int(cp1):int(cp2)] + t2, m3 * x[int(cp2):] + t3))
        return y

    xdata = np.arange(data.shape[0]) if xdata is None else xdata
    cp1_range = xdata if cp1_range is None else cp1_range
    cp2_range = xdata if cp2_range is None else cp2_range

    err_min = 1e10
    for i, cp1 in enumerate(cp1_range):
        print('Step %g / %g' % (i + 1, len(cp1_range)))
        for i, cp2 in enumerate(cp2_range):
            if cp2 > cp1:
                f = partial(piecewise, cp1=cp1, cp2=cp2)
                popt, pcov = optimize.curve_fit(f, xdata, data, [1, 0, 0, 0])
                y = f(xdata, *popt)
                err = np.sqrt(((y - data) ** 2).mean())
                if err < err_min:
                    err_min = err
                    m1, t1, m2, m3 = popt[0], popt[1], popt[2], popt[3]
                    param_min = (m1, t1, m2, m3, cp1, cp2)

    m1, t1, m2, m3, cp1, cp2 = param_min[0], param_min[1], param_min[2], param_min[3], param_min[4], param_min[5]
    t2 = xdata[int(cp1)] * (m1 - m2) + t1
    t3 = xdata[int(cp2)] * (m2 - m3) + t2
    cp1_ind = list(xdata).index(cp1)
    cp2_ind = list(xdata).index(cp2)
    p1 = linregress(xdata[:cp1], data[:cp1]).pvalue
    p2 = linregress(xdata[cp1:cp2], data[cp1:cp2]).pvalue
    p3 = linregress(xdata[cp2:], data[cp2:]).pvalue

    if verbose:
        print('\nParameters:')
        print('m1 = %.8f (p = %.1E)' % (m1, p1))
        print('t1 = %.4f' % t1)
        print('m2 = %.8f (p = %.1E)' % (m2, p2))
        print('t2 = %.4f' % t2)
        print('m3 = %.8f (p = %.1E)' % (m3, p3))
        print('t3 = %.4f' % t3)
        print('Changepoint 1 = %g' % cp1)
        print('Changepoint 2 = %g' % cp2)

    if do_plot:
        age_corr = True
        corr_str = 'age-corrected' if age_corr else 'not age-corrected'
        atlas_score = 0.70211 if age_corr else 0.68570
        normalized = False
        norm_str = 'normalized' if normalized else 'unnormalized'
        sns.set_context("notebook", font_scale=1.2)
        fig, ax = plt.subplots(1)
        st = fig.suptitle('SpectralClustering: %s, %s' % (corr_str, norm_str))
        ax.plot(xdata, data, '-', color='g')
        ax.plot(xdata[:cp1_ind+1], m1*xdata[:cp1_ind+1]+t1, 'k-', lw=2)
        ax.plot(xdata[cp1_ind:cp2_ind], m2*xdata[cp1_ind:cp2_ind]+t2, 'k-', lw=2)
        ax.plot(xdata[cp2_ind:], m3*xdata[cp2_ind:]+t3, 'k-', lw=2)
        ax.axhline(y=atlas_score, color='red', linestyle='--', linewidth=0.7)
        ax.set_xlim(0, 1500)
        ax.set_ylim(0.64, 0.74)
        ax.set_ylabel('balanced accuracy')
        ax.set_xlabel('n clusters')
        fig.savefig(os.path.join(DESKTOP, 'changepoints_spectral_%s_%s_.png' % (corr_str, norm_str)))
    return cp1, cp1_ind, cp2, cp2_ind, m1, t1, m2, t2, m3, t3, p1, p2, p3, xdata

if __name__ == '__main__':
    data = np.load(os.path.join(RESULT_DIR, 'spectral_highres_corr_unnormalized/spectral_highres_corr_unnormalized.npy'))
    x, y = data.T
    f = interp1d(x, y)
    x_new = np.arange(1, np.max(x) + 1)
    data_ = f(x_new)

    cp1_range = range(50)
    cp2_range = range(50, 500)
    # for testing
    # cp1_range = range(35, 45)
    # cp2_range = range(270, 280)
    changepoint2(data_, xdata=x_new, cp1_range=cp1_range, cp2_range=cp2_range, do_plot=True)
