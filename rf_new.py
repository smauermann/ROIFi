import json
import os
import warnings
from time import localtime, strftime
import sys

warnings.simplefilter("ignore", UserWarning)
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import numpy.ma as ma
from mpl_toolkits.mplot3d import Axes3D
from nilearn.plotting import plot_stat_map
from scipy.ndimage.measurements import label
from scipy.stats import pearsonr
from sklearn.cluster import spectral_clustering
from sklearn.feature_extraction.image import grid_to_graph
from roi_classifier import roi_classify
try:
    import cPickle as pickle
except:
    import pickle
from utils.load_data import load_structural


class ROIFinderBaseClass:
    # 3d directions to find neighbor voxels
    DIRECTIONS = dict(front=np.array([-1, 0, 0]), back=np.array([1, 0, 0]),
                      top=np.array([0, -1, 0]), bottom=np.array([0, 1, 0]),
                      left=np.array([0, 0, -1]), right=np.array([0, 0, 1]))
    # directions for reaching every adjacent voxel around the center,
    # voxels covered in DIRECTIONS are excluded
    MORE_DIRECTIONS = dict(top1=np.array([0, -1, -1]),
                           top2=np.array([0, -1, 1]),
                           bot1=np.array([0, 1, -1]),
                           bot2=np.array([0, 1, 1]),
                           front1=np.array([-1, 0, -1]),
                           front2=np.array([-1, -1, -1]),
                           front3=np.array([-1, -1, 0]),
                           front4=np.array([-1, -1, 1]),
                           front5=np.array([-1, 0, 1]),
                           front6=np.array([-1, 1, 1]),
                           front7=np.array([-1, 1, 0]),
                           front8=np.array([-1, 1, -1]),
                           back1=np.array([1, 0, -1]),
                           back2=np.array([1, -1, -1]),
                           back3=np.array([1, -1, 0]),
                           back4=np.array([1, -1, 1]),
                           back5=np.array([1, 0, 1]),
                           back6=np.array([1, 1, 1]),
                           back7=np.array([1, 1, 0]),
                           back8=np.array([1, 1, -1]))
    ALL_DIRECTIONS = {**DIRECTIONS, **MORE_DIRECTIONS}
    # number of CPUs for multiprocessing
    N_CPUS = os.cpu_count()
    ATLAS = dict(aal=os.path.join(os.getcwd(), 'r5_aal.nii.gz'))

    def __init__(self, volumes=None, mask_img=None, random_seed=None,
                 normalize_volumes=False, verbose=None):
        # mechanism to suppress output (0 = no output, 1 = medium, 2 = max)
        self.verbose = 1 if verbose is None else verbose
        if self.verbose >= 1:
            print('\n##### %s #####' % self.__str__())
        # following None value variables get actual values in the run() method
        # arbitrary threshold for pearson coefficient
        self.r_threshold = None
        # arbitrary threshold for p-value of pearson test
        self.p_threshold = None
        # filter out clusters smaller than that
        self.min_cluster_size = None

        # set random seed if needed
        if random_seed is not None:
            np.random.seed(random_seed)

        # check dtype of provided volumes, generate toy data if needed
        if volumes is not None:
            # path to nifti images
            if isinstance(volumes, str):
                    self.volumes = []
                    self.get_images_from_directory(path_to_dir=volumes)
            # check if volumes are provided as 4d array
            elif isinstance(volumes, np.ndarray):
                n_subjects = volumes.shape[0]
                self.volumes = [volumes[s, ...] for s in range(n_subjects)]
            # list as argument
            elif isinstance(volumes, list):
                # check if list carries arrays
                if all(isinstance(x, np.ndarray) for x in volumes):
                    self.volumes = volumes
                # lsit of strings, eg filenames
                elif all(isinstance(x, str) for x in volumes):
                    self.volumes = []
                    self.get_images_from_directory(file_names=volumes)
            # when passing int, toydata with dimension of int is generated
            elif isinstance(volumes, int):
                self.volumes = self.make_toy_data(dim=volumes)
        else:
            self.volumes = []
            self.load_images_and_mask()
            mask_img = self.mask
            #self.volumes = self.make_toy_data()

        # shape of the first volume, assumes all volumes have same shape
        self.volume_shape = self.volumes[0].shape

        # masking
        if mask_img is not None:
            self.mask = self.load_img_to_array(mask_img)
            self.mask_path = mask_img
            self.apply_masking()
        else:
            self.mask = None
            self.mask_path = None

        # normalize to zero mean (subtract individual means from arrays)
        if normalize_volumes:
            self.volumes_normalized = self.normalize_volumes()
        self.normalized = normalize_volumes

        # number of subjects
        self.n_subjects = len(self.volumes)
        # number of non-nan voxels
        self.n_voxels = self.volumes[0].size - np.isnan(self.volumes[0]).sum()
        # num of clusters detected or number of cluster required for spectral
        # clustering approach
        self.cluster_count = 0
        # mean size of clusters
        self.mean_cluster_size = 0
        # couple of arrays that contain cluster information
        self.cluster_array_bool = np.full(self.volume_shape, False, dtype=bool)
        self.cluster_array_labelled = None
        # .nii image of cluster array
        self.cluster_img = None
        # dict containing information about clusters and metadata of analysis
        self.metadata = dict()
        # figure holding glassbrain or scatter plot
        self.fig = None
        # overlap with clusters and reference atlas
        self.atlas = None  # atlas to be compared with
        self.atlas_coverage = None  # actual percentage of overlap
        # array containing means per subject, per cluster
        self.cluster_means = None
        self.classification_score = None

    def make_toy_data(self, dim=8):
        if self.verbose >= 1:
            print('No data provided, toy data will be generated!')
        n_subjects = 100
        sub_dim = dim // 2
        # noise array for all samples
        noise = np.random.rand(dim, dim, dim, n_subjects)
        # add a signal to each of the 8 sub cubes here
        noise[:sub_dim, :sub_dim, :sub_dim] += np.random.rand(n_subjects)
        # noise[:sub_dim, sub_dim:, :sub_dim] += np.random.rand(n_subjects)
        # noise[:sub_dim, sub_dim:, sub_dim:] += np.random.rand(n_subjects)
        # noise[:sub_dim, :sub_dim, sub_dim:] += np.random.rand(n_subjects)
        # noise[sub_dim:, :sub_dim, :sub_dim] += np.random.rand(n_subjects)
        # noise[sub_dim:, sub_dim:, :sub_dim] += np.random.rand(n_subjects)
        noise[sub_dim:, sub_dim:, sub_dim:] += np.random.rand(n_subjects)
        # noise[sub_dim:, :sub_dim, sub_dim:] += np.random.rand(n_subjects)

        # make list of 3d arrays
        return [noise[..., s] for s in range(n_subjects)]

    @staticmethod
    def load_img_to_array(img_path):
        img = nib.load(img_path)
        img_data = np.array(img.get_data())
        img.uncache()
        return img_data

    def get_images_from_directory(self, path_to_dir=None, file_names=None):
        if self.verbose >= 1:
            print('Loading images from disk ...')
        # if list of file names was provided get only these
        if file_names is not None:
            if all(os.path.isfile(x) for x in file_names):
                for name in file_names:
                    img_data = self.load_img_to_array(name)
                    self.volumes.append(img_data)
                n_images = len(file_names)
            else:
                raise FileNotFoundError('Some provided files do NOT exist!')
            self.affine = nib.load(file_names[0]).affine
        # get all nii-files from directory
        elif path_to_dir is not None:
            if os.path.isdir(path_to_dir):
                for img in os.listdir(path_to_dir):
                    if img.endswith('.nii'):
                        img_path = os.path.join(path_to_dir, img)
                        img_data = self.load_img_to_array(img_path)
                        self.volumes.append(img_data)
                n_images = len(os.listdir(path_to_dir))
            else:
                raise FileNotFoundError('%s is not a valid path!' % path_to_dir)
            self.affine = nib.load(img_path).affine
        if self.verbose >= 1:
            print('Loaded %d images!' % n_images)

    def apply_masking(self):
        if self.verbose >= 1:
            print('Applying mask ...')
        # check if mask and volumes have same shape
        if self.volume_shape == self.mask.shape:
            # switch ones and zeros because np.MaskedArray masks values if true (=1)
            # I think neuroimaging masks indicate invalid values with false (=0)
            mask_switched = self.mask ^ 1  # bitwise XOR
            # loop over volumes and mask them
            for i, v in enumerate(self.volumes):
                self.volumes[i] = ma.array(v, mask=mask_switched,
                                           fill_value=np.nan).filled()
        else:
            raise ValueError('Mask.shape %s and Volumes.shape %s must be identical!'
                             % (self.mask.shape, self.volume_shape))

    def normalize_volumes(self):
        if self.verbose >= 1:
            print("Normalizing volumes ...")
        # normalize all volumes to zero mean
        if self.mask is not None:
            volumes_normalized = [v - np.nanmean(v) for v in self.volumes]
        else:
            volumes_normalized = [v - np.mean(v) for v in self.volumes]
        return volumes_normalized

    def draw_brain_map(self):
        cmap = plt.get_cmap('Accent')
        self.fig = plt.figure('brain_map')
        plot_stat_map(self.cluster_img, cut_coords=(0, 0, 0), output_file=None,
                      display_mode='ortho', colorbar=False, figure=self.fig,
                      axes=None, title=None, threshold=0.1, annotate=True,
                      draw_cross=False, black_bg='auto', symmetric_cbar="auto",
                      dim=True, vmax=None, cmap=cmap)

    def scatter_clusters(self):
        clusters = self.cluster_array_labelled
        self.fig = plt.figure(num='scatter', figsize=(10, 10))
        ax = self.fig.add_subplot(111, projection='3d')
        y, z, x = np.where(clusters != 0)
        cluster_labels = list(np.sort(clusters[y, z, x]))

        ax.scatter(x, y, z, zdir='z', c=cluster_labels)
        ax.set_xlabel('Front (x)')
        ax.set_xlim(0, self.volume_shape[2])
        # ax.invert_xaxis()
        ax.set_ylabel('Side (y)')
        ax.set_ylim(0, self.volume_shape[0])
        ax.set_zlabel('Rows (z)')
        ax.set_zlim(0, self.volume_shape[1])
        ax.invert_zaxis()

    def save_results(self, output_path):
        timestamp = strftime('%d%m%y_%H%M', localtime())
        if self.__str__() == 'SpectralClustering':
            method = 'spectral_'
            parameters = '%sclusters_%s_' % (self.cluster_count,
                                             self.distance_measure)
        elif self.__str__() == 'PearsonMerger':
            method = 'pearson_'
            if self.p_threshold is not None:
                parameters = 'rt%s_pt%s_' % (self.r_threshold, self.p_threshold)
            else:
                parameters = 'rt%s_' % (self.r_threshold)
        # make results directory
        result_dir = os.path.join(output_path, method + parameters + timestamp)
        try:
            os.makedirs(result_dir)
        except OSError:
            i = 1
            while os.path.isdir(result_dir):
                result_dir += '({})'.format(i)
                i += 1
            os.makedirs(result_dir)

        if self.cluster_means is not None:
            cluster_means_file = os.path.join(result_dir, 'cluster_means')
            np.save(cluster_means_file, self.cluster_means)
        # save figures is plotted
        if self.fig is not None:
            plt.savefig(os.path.join(result_dir, 'cluster_plot.png'))
        # save metadata dict as json
        metadata_file = os.path.join(result_dir, 'metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, sort_keys=True, indent=4)
        # save clusters as nifti image
        nib.save(self.cluster_img, os.path.join(result_dir, 'cluster_img.nii.gz'))
        # pickle affine and cluster_means (latter if created)
        # affine_file = os.path.join(result_dir, 'affine.pickle')
        # with open(affine_file, mode='wb') as f:
        #     pickle.dump(self.affine, f)
        if self.verbose >= 1:
            print('Saved in %s' % (result_dir))

    def clusters_vs_atlas(self, clusters, atlas):
        if self.verbose >= 1:
            print('Computing overlap of clusters and atlas!')
        # compare obtained clustering to reference atlas
        if atlas in self.ATLAS:
            self.atlas = self.ATLAS[atlas]
        else:
            self.atlas = atlas
        arrays = [clusters, self.atlas]
        for i, a in enumerate(arrays):
            # load data if needed and make it np.array
            if isinstance(a, nib.Nifti1Image):
                data = np.array(a.get_data())
            elif isinstance(a, np.ndarray):
                data = a
            else:
                data = self.load_img_to_array(a)
            # flatten both arrays, remove np.nans
            # change dtype to save memory, uint8 should be fine here (0...255)c
            arrays[i] = data.astype(np.uint8)
        # find indices where both arrays are not zero
        not_zero_ids = np.where(np.logical_and(arrays[0] != 0, arrays[1] != 0))
        # check for every voxel pair if in same cluster, 0 if yes otherwise 1
        for i, a in enumerate(arrays):
            # only elements that are not zero in both arrays
            nz = a[not_zero_ids]
            # all possible voxel pairs
            cart = self.cartesian(nz, nz)
            # check for each pair if in same cluster
            arrays[i] = np.equal(cart[:, 0], cart[:, 1])
        percentage_coverage = np.mean(arrays[0] == arrays[1]) * 100
        if self.verbose >= 1:
            print('Atlas - clusters overlap: %.2f %% ' % (percentage_coverage))
        return percentage_coverage

    @staticmethod
    def cartesian(x, y):
        return np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])

    def make_cluster_means(self, other_clusters=None):
        if self.verbose >= 1:
            print('Computing cluster means ...')
        if other_clusters is None:
            clusters = self.cluster_array_labelled
            n_clusters = self.cluster_count
            cluster_ids = range(1, n_clusters + 1)
        else:
            clusters = other_clusters.astype('float')
            # check if zeros in array and replace with nan
            clusters[clusters == 0] = np.nan
            # in case cluster ids are not numbered sequentially,
            # ie if there are gaps in the numbering
            cluster_ids = list(np.unique(clusters[~np.isnan(clusters)]))
            n_clusters = len(cluster_ids)
        # store mean values per cluster, per subject in an array
        # (n_subjects x n_clusters)
        self.cluster_means = np.empty((self.n_subjects, n_clusters))
        for s, c in np.ndindex(self.cluster_means.shape):
            c_idx = np.where(clusters == cluster_ids[c])
            self.cluster_means[s, c] = np.mean(self.volumes[s][c_idx])

    def do_classification(self, estimator, labels):
        if self.cluster_means is None:
            self.make_cluster_means()
        if self.verbose >= 1:
            print('\nPerforming Classification ...')
        if labels is None:
            print('No labels for classification provided!')
        else:
            self.classification_score = roi_classify(self.cluster_means, labels,
                                                     self.n_subjects,
                                                     estimator=estimator)
            return (self.cluster_count, self.classification_score)

    def load_images_and_mask(self, typ='all'):
        RESAMPLING = 5
        HOME = os.path.expanduser("~")
        ROOT = 'Google_Drive/Master_Thesis/ROI_project'
        MASKS = {0: 'alc_P2_mask.nii', 3: 'r3alc_P2_mask.nii', 5: 'r5alc_P2_mask.nii'}
        TYPEDICT = {'all': None, 'patients': 1, 'controls': 0}

        img_files, labels, subjects = load_structural(project=2, smoothing=8,
                                                      resampling=RESAMPLING,
                                                      type=TYPEDICT[typ],
                                                      corrected=True)

        self.get_images_from_directory(file_names=img_files)
        self.mask = os.path.join(HOME, ROOT, MASKS[RESAMPLING])

    def run(self):
        raise NotImplementedError

    def find_clusters(self):
        raise NotImplementedError

    def update_metadata(self):
        raise NotImplementedError

    def __str__(self):
        return self.__class__.__name__


class SpectralClustering(ROIFinderBaseClass):
    def __init__(self, volumes=None, mask_img=None, random_seed=None,
                 normalize_volumes=False, verbose=None):
        super().__init__(volumes=volumes, mask_img=mask_img,
                         random_seed=random_seed,
                         normalize_volumes=normalize_volumes, verbose=verbose)

    def run(self, output_path=None, n_clusters=100, draw=None,
            distance_measure='correlation', atlas=None,
            cluster_means=False):
        # important, cluster_count here defines the number of clusters fitted
        # onto the data
        self.cluster_count = n_clusters
        # either 'correlation' or 'rbf_euclidian'
        self.distance_measure = distance_measure

        # do the actual work
        self.find_clusters()

        if atlas is not None:
            self.atlas_coverage = self.clusters_vs_atlas(self.cluster_img, atlas)

        self.update_metadata()

        if self.verbose >= 2:
            for key in sorted(self.metadata):
                print(key, self.metadata[key])

        if draw is not None:
            if draw == 'scatter':
                self.scatter_clusters()
            elif draw == 'brain_map' or draw == 'brainmap':
                self.draw_brain_map()

        if cluster_means:
            self.make_cluster_means()

        if output_path is not None:
                self.save_results(output_path)

    def find_clusters(self):
        if self.verbose >= 1:
            print('\nFinding ROIs ...')
            print('parameters:')
            print('    n_clusters: %s\n    distance_measure: %s'
                  % (self.cluster_count, self.distance_measure))
        if self.normalized:
            volumes = self.volumes_normalized
        else:
            volumes = self.volumes
        # Define a spatial model
        s = self.volume_shape
        # make a graph representation of the volume
        connectivity = grid_to_graph(s[0], s[1], s[2], mask=self.mask).tocsr()
        # transform volumes to obtain array of n_voxels x n_subjects
        vs = np.concatenate([x[~np.isnan(x)] for x in volumes]).reshape(self.n_voxels, self.n_subjects)
        # pairs of all adjacent voxels
        i, j = connectivity.nonzero()
        if self.distance_measure == 'rbf_euclidian':
            # rbf kernel on euclidian distance, scaled by sigma
            # as in Thirion et al. (2014)
            connectivity.data = self.rbf_euclidian(vs[i], vs[j])
        elif self.distance_measure == 'correlation':
            # np.squeeze removes single axis to make array 1D, compressed sparse
            # matrix needs this
            # scaled correlation bounded to [0,1]
            connectivity.data = (1 + self.corr2_coeff(vs[i], vs[j])) / 2
        # spectral clustering on the weighted connectivity dense matrix
        # arpack is a performant FORTRAN eigen vector problem solver
        labels = spectral_clustering(connectivity,
                                     n_clusters=self.cluster_count,
                                     eigen_solver='arpack')
        # labels start with 0, add 1 to match the style in this project
        labels += 1
        # fill cluster array with labels from spectral clustering
        self.cluster_array_labelled = np.full(self.volume_shape, np.nan)
        self.cluster_array_labelled[self.mask == 1] = labels
        self.cluster_img = nib.Nifti1Image(self.cluster_array_labelled, self.affine)
        if self.verbose >= 1:
            print('\nAnalysis completed!')

    @staticmethod
    def rbf_euclidian(x, y):
        # scale factor for RBF kernel
        sigma = np.sum((x - y) ** 2, axis=1).mean()
        # RBF kernel over all adjacent voxels
        return np.exp(- np.sum((x - y) ** 2, axis=1) / (2 * sigma))

    @staticmethod
    def corr2_coeff(x, y):
        n = x.shape[1]
        # Rowwise mean of input arrays
        mean_x = x.mean(axis=1)[:, np.newaxis]
        mean_y = y.mean(axis=1)[:, np.newaxis]
        # rowwise std deviation
        std_x = x.std(axis=1, ddof=n - 1)[:, np.newaxis]
        std_y = y.std(axis=1, ddof=n - 1)[:, np.newaxis]
        # row wise covariance
        cov = (np.sum((x - mean_x) * (y - mean_y), axis=1) / n)[:, np.newaxis]
        # Finally get corr coeff
        return np.squeeze(cov / (std_x * std_y))

    def rank_clusters(self):
        # get mean correlation for each cluster
        pass

    def update_metadata(self):
        # erase all entries from previous runs
        self.metadata = dict()
        # predefined number of clusters
        self.metadata['cluster_count'] = self.cluster_count
        # similarity measure used for clustering
        self.metadata['distance_measure'] = self.distance_measure
        self.metadata['normalized'] = self.normalized
        if self.mask is not None:
            self.metadata['mask'] = self.mask_path
        if self.atlas_coverage is not None:
            self.metadata['atlas'] = self.atlas
            self.metadata['atlas_coverage'] = self.atlas_coverage
        # loop over clusters
        for i in range(self.cluster_count):
            n_cluster = i + 1
            cluster_size = np.where(self.cluster_array_labelled == n_cluster)[0].size
            cluster_id = "size_cluster_%03d" % (n_cluster)
            self.metadata[cluster_id] = cluster_size


class PearsonMerger(ROIFinderBaseClass):
    def __init__(self, volumes=None, mask_img=None, random_seed=None,
                 normalize_volumes=False, verbose=None):
        super().__init__(volumes=volumes, mask_img=mask_img,
                         random_seed=random_seed,
                         normalize_volumes=normalize_volumes, verbose=verbose)

    def run(self, output_path=None, r_threshold=None, p_threshold=None,
            min_cluster_size=None, draw=None, auto_threshold=False,
            atlas=None, cluster_means=False):
        # arbitrary threshold for pearson coefficient
        if r_threshold is not None:
            self.r_threshold = r_threshold
            auto_threshold = False
        # arbitrary threshold for p-value of pearson test
        if p_threshold is not None:
            self.p_threshold = p_threshold
        # filter out clusters smaller than that
        if min_cluster_size is not None:
            self.min_cluster_size = min_cluster_size

        # only estimate threshold when no r_threshold is provided
        if auto_threshold or self.r_threshold is None:
            self.auto_threshold()
        # do the work
        self.find_clusters()
        if atlas is not None:
            self.atlas_coverage = self.clusters_vs_atlas(self.cluster_img, atlas)
        self.update_metadata()
        if self.verbose >= 1:
            print('\nAnalysis completed, detected %s cluster(s)!' % self.cluster_count)
            if self.verbose >= 2:
                for key in sorted(self.metadata):
                    print(key, self.metadata[key])
        if draw is not None:
            if draw == 'scatter':
                self.scatter_clusters()
            elif draw == 'brain_map' or draw == 'brainmap':
                self.draw_brain_map()
        if cluster_means:
            self.make_cluster_means()
        if output_path is not None:
            self.save_results(output_path)

    def auto_threshold(self):
        if self.verbose >= 1:
            print('Autothresholding r ...')
        all_correlations = []
        # get all correlation values and make histogram
        for index in np.ndindex(self.volume_shape):
            # print("\rProcessing %s / %s" % (current_voxel, total_voxels), end="")
            # check that element is not masked
            if not np.isnan(self.volumes[0][index]):
                center_voxels = self.get_voxels(index)
                # get neighboring voxel indices
                neighbor_indices = self.get_neighbor_indices(index,
                                                             all_directions=False)
                # extract voxel values for neighbors
                neighbor_voxels = self.get_neighbor_voxels(neighbor_indices)
                for val in neighbor_voxels.values():
                    r, _ = pearsonr(center_voxels, val)
                    all_correlations.append(r)
        all_correlations = np.array(all_correlations)
        mean_r = np.mean(all_correlations)
        percent = 90
        percentile_r = np.around(np.percentile(all_correlations, percent),
                                 decimals=2)
        self.r_threshold = percentile_r if percentile_r < 1.0 else 0.99
        # plot histogram
        if self.verbose >= 1:
            print('Threshold of r set to %s' % self.r_threshold)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            hist, bin_edges = np.histogram(all_correlations)
            width = 0.7 * (bin_edges[1] - bin_edges[0])
            center = (bin_edges[:-1] + bin_edges[1:]) / 2
            ax.bar(center, hist, align='center', width=width)
            ymin, ymax = ax.get_ylim()
            # plot mean
            ax.axvline(mean_r, color='black', linestyle='dashed')
            ax.text(mean_r, ymax * 0.9, 'mean=%.2f' % mean_r, rotation=-90)
            # plot percentile
            ax.axvline(percentile_r, color='black', linestyle='dashed')
            ax.text(percentile_r, ymax * 0.9, '%sth=%s' % (percent, percentile_r),
                    rotation=-90)
            # plot cosmetics and labels
            ax.set_title('Correlation distribution in sample')
            ax.set_xlabel('Pearson Coefficients')
            ax.set_ylabel('n')

    def find_clusters(self):
        if self.verbose >= 1:
            print('\nFinding ROIs ...')
            print('parameters:')
            print('    r_threshold: %s\n    p_threshold: %s\n    min_cluster_size: %s'
                  % (self.r_threshold if self.r_threshold is not None else 'auto',
                     self.p_threshold, self.min_cluster_size))
        total_voxels = self.n_voxels
        current_voxel = 1
        # reset this array in case several runs on same instance of this class
        self.cluster_array_bool = np.full(self.volume_shape, False, dtype=bool)
        # iterates over all indices of the volume, returns tuples
        for index in np.ndindex(self.volume_shape):
            if self.verbose == 2:
                print("\rProcessing %s / %s" % (current_voxel, total_voxels),
                      end="")
            # check that element is not masked
            if not np.isnan(self.volumes[0][index]):
                self.compute_clusters(index)
            current_voxel += 1

        self.label_clusters()

    def compute_clusters(self, center_index):
        # collect center voxel from all volumes
        center_voxels = self.get_voxels(center_index)
        # get neighboring voxel indices
        neighbor_indices = self.get_neighbor_indices(center_index,
                                                     all_directions=False)
        # extract voxel values for neighbors
        neighbor_voxels = self.get_neighbor_voxels(neighbor_indices)
        # Pearson with surrounding voxels
        correlation_scores = self.compute_correlation(center_voxels,
                                                      neighbor_voxels)
        if correlation_scores is not None:
            # write changes to the cluster_array
            self.cluster_array_bool[center_index] = True

    def compute_correlation(self, center_voxels, neighbor_voxels):
        correlation_scores = dict()
        for key, val in neighbor_voxels.items():
            r, p = pearsonr(center_voxels, val)
            # filter according to threshold levels for r and p
            # p values are not entirely reliable according to documentation
            # only for datasets larger than 500
            if r >= self.r_threshold:
                if self.p_threshold is not None:
                    if p <= self.p_threshold:
                        correlation_scores[key] = r, p
                else:
                    correlation_scores[key] = r, p
        # check if dict is not empty
        if correlation_scores:
            return correlation_scores
        else:
            return None

    def label_clusters(self):
        # by default only fully connected voxels and no diagonal connections
        # if needed structure must be passed, eg structure = np.ones((3,3,3))
        self.cluster_array_labelled, self.cluster_count = label(self.cluster_array_bool)
        # filter out clusters smaller than threshold
        if self.min_cluster_size is not None:
            # loop over clusters
            for i in range(self.cluster_count):
                n_cluster = i + 1
                cluster_ids = np.where(self.cluster_array_labelled == n_cluster)
                cluster_size = cluster_ids[0].size
                # if cluster below limit set it to np.nan in cluster_array
                if cluster_size < self.min_cluster_size:
                    self.cluster_array_labelled[cluster_ids] = 0
            self.cluster_array_labelled, self.cluster_count = label(self.cluster_array_labelled)

        if self.cluster_count == 0:
            raise NoClustersError('Exiting PearsonMerger: parameters too strict, no clusters detectable!')
        # overwrite zeros with nan for plotting
        zeros_ids = np.where(self.cluster_array_labelled == 0)
        # doesnt work on int array, thus convert to float type
        self.cluster_array_labelled = self.cluster_array_labelled.astype('float')
        self.cluster_array_labelled[zeros_ids] = np.nan
        self.cluster_img = nib.Nifti1Image(self.cluster_array_labelled,
                                           self.affine)

    def get_voxels(self, index):
        if self.normalized:
            return np.array([v[index] for v in self.volumes_normalized])
        else:
            return np.array([v[index] for v in self.volumes])

    def get_neighbor_indices(self, center_index, all_directions=False):
        neighbor_indices = dict()
        directions = (self.DIRECTIONS if all_directions is False
                      else self.ALL_DIRECTIONS)
        for key, val in directions.items():
            new_index = tuple(center_index + val)  # new_index would else be np.array
            # check if index is valid and corresponding element is not np.nan
            if self.is_valid_index(new_index):
                neighbor_indices[key] = new_index
        # returns dict with valid indices of adjacent cells
        return neighbor_indices

    def is_valid_index(self, index):
        # checks if index is valid and corresponding element is not np.nan
        for i, n in enumerate(index):
            if (n < 0) or (n > self.volume_shape[i] - 1):
                return False
        if np.isnan(self.volumes[0][index]):
            return False
        return True

    def get_neighbor_voxels(self, indices):
        neighbor_voxels = dict()
        for key, val in indices.items():
            neighbor_voxels[key] = self.get_voxels(val)
        return neighbor_voxels

    def mean_cluster_correlations(self, clusters):
        if self.verbose >= 1:
            print('Computing cluster correlation maps ...')
        # check what dtype clusters are
        if isinstance(clusters, np.ndarray):
            cluster_data = clusters
        elif isinstance(clusters, nib.Nifti1Image):
            cluster_data = np.array(clusters.get_data())
        elif os.path.isfile(clusters):
            cluster_data = self.load_img_to_array(clusters)

        # check if clusters and self.volumes have same shape
        if self.volume_shape != cluster_data.shape:
            raise ValueError('Volumes and cluster must have same shape!')

        # now we have clusters as an array
        cluster_ids = np.unique(cluster_data)
        cluster_ids = cluster_ids[~np.isnan(cluster_ids)].astype('int')
        cluster_correlation_map = np.empty((len(cluster_ids), 2))
        cluster_correlation_map[:, 0] = cluster_ids
        # now iterate over array ids and compute correlations of all
        # adjacent voxels within the array

        for i, c in enumerate(cluster_ids):
            # get indices of all cluster voxels
            c_indices = np.where(cluster_data == c)
            x, y, z = list(c_indices[0]), list(c_indices[1]), list(c_indices[2])
            cluster_coords = list(zip(x, y, z))
            # iterate over all voxels in cluster
            cluster_correlation = []
            for coord in cluster_coords:
                center_voxels = self.get_voxels(coord)
                neighbor_indices = self.get_neighbor_indices(coord)
                # sweep out neigbors not in the current cluster
                neighbor_indices = [v for k, v in neighbor_indices.items() if v in cluster_coords]
                neighbor_voxels = [self.get_voxels(x) for x in neighbor_indices]
                voxel_correlation = []
                for voxel in neighbor_voxels:
                    r, _ = pearsonr(center_voxels, voxel)
                    voxel_correlation.append(r)
                mean_voxel_correlation = np.mean(np.array(voxel_correlation))
                cluster_correlation.append(mean_voxel_correlation)
            mean_cluster_correlation = np.mean(np.array(cluster_correlation))
            cluster_correlation_map[i, 1] = mean_cluster_correlation

        return cluster_correlation_map

    def update_metadata(self):
        # erase all entries from previous runs
        self.metadata = dict()
        self.metadata['r_thresh'] = self.r_threshold
        self.metadata['p_thresh'] = self.p_threshold
        self.metadata['cluster_count'] = self.cluster_count
        self.metadata['normalized'] = self.normalized
        if self.mask is not None:
            self.metadata['mask'] = self.mask_path
        if self.atlas_coverage is not None:
            self.metadata['atlas'] = self.atlas
            self.metadata['atlas_coverage'] = self.atlas_coverage
        # filter out clusters smaller than threshold
        if self.min_cluster_size is not None:
            self.metadata['min_cluster_size'] = self.min_cluster_size
        # loop over clusters
        cum_cluster_size = 0
        for i in range(self.cluster_count):
            n_cluster = i + 1
            cluster_size = np.where(self.cluster_array_labelled == n_cluster)[0].size
            cum_cluster_size += cluster_size
            cluster_id = "size_cluster_%03d" % (n_cluster)
            self.metadata[cluster_id] = cluster_size
        self.metadata['mean_cluster_size'] = cum_cluster_size / self.cluster_count


class NoClustersError(Exception):
    def __init__(self, message):
        self.message = message
