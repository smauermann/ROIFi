import os
from time import localtime, strftime

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import numpy.ma as ma
from mpl_toolkits.mplot3d import Axes3D
from nilearn.plotting import plot_glass_brain
from scipy.ndimage.measurements import label
from scipy.stats import pearsonr
from sklearn.cluster import spectral_clustering
from sklearn.feature_extraction.image import grid_to_graph
from scipy.sparse import dia_matrix

try:
    import cPickle as pickle
except:
    import pickle


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
            self.volumes = self.make_toy_data()

        # shape of the first volume, assumes all volumes have same shape
        self.volume_shape = self.volumes[0].shape

        # masking
        if mask_img is not None:
            self.mask = self.load_img_to_array(mask_img)
            self.apply_masking()
        else:
            self.mask = None

        # normalize to zero mean (subtract individual means from arrays)
        if normalize_volumes:
            self.normalize_volumes()

        # number of subjects
        self.n_subjects = len(self.volumes)
        # num of clusters detected or number of cluster required for spectral
        # clustering approach
        self.cluster_count = 0
        # couple of arrays that contain cluster information
        self.cluster_array_bool = np.full(self.volume_shape, False, dtype=bool)
        self.cluster_array_labelled = None
        # .nii image of cluster array
        self.cluster_img = None
        # dict containing information about clusters and metadata of analysis
        self.metadata = dict()

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

    def get_voxels(self, index):
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

    @staticmethod
    def load_img_to_array(img_path):
        img = nib.load(img_path)
        img_data = img.get_data()
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
            else:
                raise FileNotFoundError('%s is not a valid path!' % path_to_dir)
            self.affine = nib.load(img_path).affine

    def apply_masking(self):
        if self.verbose >= 1:
            print("Applying mask ...")
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
            raise ValueError('Mask.shape %s and Volumes.shape %s must be \
                             identical!' % (self.mask.shape, self.volume_shape))

    def normalize_volumes(self):
        if self.verbose >= 1:
            print("Normalizing volumes ...")
        # normalize all volumes to zero mean
        if self.mask is not None:
            for v in self.volumes:
                v -= np.nanmean(v)
        else:
            for v in self.volumes:
                v -= np.mean(v)

    def draw_glass_brain(self):
        if self.verbose >= 1:
            print("Drawing glass brain ...")
        cmap = plt.get_cmap('Accent')
        plot_glass_brain(self.cluster_img, output_file=None, display_mode='ortho',
                         colorbar=True, figure=None, axes=None, title=None,
                         threshold=0.1, annotate=True, black_bg=False,
                         cmap=cmap, alpha=0.7, vmin=None, vmax=None,
                         plot_abs=True, symmetric_cbar=False)

    def scatter_clusters(self):
        if self.verbose >= 1:
            print("Scatter plotting clusters ...")
        clusters = self.cluster_array_labelled
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
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
            method = 'spectral_clustering_'
            parameters = ''
        elif self.__str__() == 'PearsonMerger':
            method = 'pearson_merger_'
            if self.p_threshold is not None:
                parameters = 'rt%s_pt%s_' % (self.r_threshold, self.p_threshold)
            else:
                parameters = 'rt%s_' % (self.r_threshold)
        result_dir = os.path.join(output_path, method + parameters + timestamp)
        os.mkdir(result_dir)
        output_file = os.path.join(result_dir, 'results.pickle')
        results = (self.metadata, self.cluster_array_labelled)
        with open(output_file, 'wb') as f:
            pickle.dump(results, f)
        nib.save(self.cluster_img, os.path.join(result_dir, 'cluster_img.nii.gz'))
        plt.savefig(os.path.join(result_dir, 'cluster_plot.png'))
        if self.verbose >= 1:
            print('Saved in %s' % (result_dir))

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

    def run(self, output_path=None, n_clusters=100, draw=None):
        # important, cluster_count here defines the number of clusters fitted
        # onto the data
        self.cluster_count = n_clusters

        self.find_clusters()
        self.update_metadata()

        if self.verbose >= 1:
            print('\nAnalysis completed!')
            for key in sorted(self.metadata):
                print(key, self.metadata[key])

        if draw is not None:
            if draw == 'scatter':
                self.scatter_clusters()
            elif draw == 'glassbrain' or draw == 'glass_brain':
                self.draw_glass_brain()

        if output_path is not None:
            self.save_results(output_path)

    def find_clusters(self):
        if self.verbose >= 1:
            print('Finding ROIs ...')
        # Define a spatial model
        s = self.volume_shape
        # make a graph representation of the volume
        connectivity = grid_to_graph(s[0], s[1], s[2], mask=self.mask).tocsr()
        # transform volumes to obtain array of n_voxels x n_subjects
        for i, v in enumerate(self.volumes):
            # flattened array
            flat = v[~np.isnan(v)]
            if i == 0:
                vox_subj = flat
            else:
                vox_subj = np.vstack((vox_subj, flat))
        # transpose to get shape of vox by sub
        vs = vox_subj.T
        # pairs of all adjacent voxels
        i, j = connectivity.nonzero()
        rbf_distance = True
        correlation_distance = not rbf_distance
        # rbf kernel on euclidian distance, scaled by sigma
        # as in Thirion et al. (2014)
        if rbf_distance:
            # scale factor for RBF kernel
            sigma = np.sum((vs[i] - vs[j]) ** 2, axis=1).mean()
            # RBF kernel over all adjacent voxels
            connectivity.data = np.exp(- np.sum((vs[i] - vs[j]) ** 2, axis=1) / (2 * sigma))

        if correlation_distance:
            pass
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

    def update_metadata(self):
        self.metadata['cluster_count'] = self.cluster_count
        # filter out clusters smaller than threshold
        if self.min_cluster_size is not None:
            self.metadata['min_cluster_size'] = self.min_cluster_size
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
            min_cluster_size=None, draw=None, auto_threshold=False):
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

        self.find_clusters()
        self.update_metadata()

        if self.verbose >= 1:
            print('\nAnalysis completed!')
            for key in sorted(self.metadata):
                print(key, self.metadata[key])

        if draw is not None:
            if draw == 'scatter':
                self.scatter_clusters()
            elif draw == 'glassbrain' or draw == 'glass_brain':
                self.draw_glass_brain()

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
        percent = 95
        percentile_r = np.around(np.percentile(all_correlations, percent),
                                 decimals=2)
        self.r_threshold = percentile_r
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
            print('Finding clusters ...')
        total_voxels = (self.volume_shape[0] * self.volume_shape[1] *
                        self.volume_shape[2])
        current_voxel = 1
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

        # overwrite zeros with nan for plotting
        zeros_ids = np.where(self.cluster_array_labelled == 0)
        # doesnt work on int array, thus convert to float type
        self.cluster_array_labelled = self.cluster_array_labelled.astype(float)
        self.cluster_array_labelled[zeros_ids] = np.nan
        self.cluster_img = nib.Nifti1Image(self.cluster_array_labelled,
                                           self.affine)

    def update_metadata(self):
        self.metadata['r_thresh'] = self.r_threshold
        self.metadata['p_thresh'] = self.p_threshold
        self.metadata['cluster_count'] = self.cluster_count
        # filter out clusters smaller than threshold
        if self.min_cluster_size is not None:
            self.metadata['min_cluster_size'] = self.min_cluster_size
        # loop over clusters
        for i in range(self.cluster_count):
            n_cluster = i + 1
            cluster_size = np.where(self.cluster_array_labelled == n_cluster)[0].size
            cluster_id = "size_cluster_%03d" % (n_cluster)
            self.metadata[cluster_id] = cluster_size
