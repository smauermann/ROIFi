import os
from itertools import combinations, product
from time import localtime, strftime

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import numpy.ma as ma
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import pearsonr
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.manifold import spectral_embedding

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

    def __init__(self, volumes=None, mask_img=None, random_seed=None):
        # following None value variables get actual values in the run() method
        # arbitrary threshold for pearson coefficient
        self.r_threshold = None
        # arbitrary threshold for p-value of pearson test
        self.p_threshold = None
        # filter out clusters smaller than that
        self.min_cluster_size = None
        # number of jobs for multiprocessing
        self.n_jobs = 1

        # set random seed if needed
        if random_seed is not None:
            np.random.seed(random_seed)

        # check dtype of provided volumes, generate toy data if needed
        if volumes is not None:
            # path to nifti images
            if isinstance(volumes, str):
                if os.path.isdir(volumes):
                    self.volumes = []
                    self.get_images_from_directory(volumes)
                else:
                    raise IOError('%s is not a directory!' % volumes)
            # check if volumes are provided as 4d array
            elif isinstance(volumes, np.ndarray):
                n_subjects = volumes.shape[0]
                self.volumes = [volumes[s, ...] for s in range(n_subjects)]
            # list of 3d arrays
            elif isinstance(volumes, list):
                self.volumes = volumes
            # when passing int, toydata with dimension of int is generated
            elif isinstance(volumes, int):
                self.volumes = self.make_toy_data(dim=volumes)
        else:
            print('No data provided, toy data with default settings will be generated!')
            self.volumes = self.make_toy_data()
        # shape of the first volume, assumes all volumes have same shape
        self.volume_shape = self.volumes[0].shape
        # number of subjects
        self.n_subjects = len(self.volumes)

        # masking
        if mask_img is not None:
            self.mask = self.load_img_to_array(mask_img)
            self.apply_masking()
        else:
            self.mask = None

        # keeps track of number of different clusters
        self.cluster_count = 0
        # array that contains integers that indicate clusters
        self.cluster_array = np.full(self.volume_shape, np.nan)
        # dict containing information about clusters and metadata of analysis
        self.metadata = dict()

    def make_toy_data(self, dim=20):
        n_subjects = 100
        noise = np.random.rand(dim, dim, dim, n_subjects)
        sub_dim = dim // 2
        sub_voxels = sub_dim ** 3

        noise[:sub_dim, :sub_dim, :sub_dim] += np.random.rand(100)
        # noise[:sub_dim, sub_dim:, :sub_dim] += np.random.rand(100)
        # noise[:sub_dim, sub_dim:, sub_dim:] += np.random.rand(100)
        # noise[:sub_dim, :sub_dim, sub_dim:] += np.random.rand(100)
        # noise[sub_dim:, :sub_dim, :sub_dim] += np.random.rand(100)
        # noise[sub_dim:, sub_dim:, :sub_dim] += np.random.rand(100)
        noise[sub_dim:, sub_dim:, sub_dim:] += np.random.rand(100)
        # noise[sub_dim:, :sub_dim, sub_dim:] += np.random.rand(100)

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

    def get_images_from_directory(self, path_to_dir):
        print('Loading images from disk ...')
        # get all files from directory
        for img in os.listdir(path_to_dir):
            if img.endswith('.nii'):
                img_path = os.path.join(path_to_dir, img)
                img_data = self.load_img_to_array(img_path)
                self.volumes.append(img_data)
        if not self.volumes:
            raise FileNotFoundError('No .nii-files in %s!' % (path_to_dir))

    def apply_masking(self):
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

    def run(self, output_path=None, r_threshold=0.5, p_threshold=None,
            min_cluster_size=None, n_jobs=1):
        # arbitrary threshold for pearson coefficient
        self.r_threshold = r_threshold
        # arbitrary threshold for p-value of pearson test
        if p_threshold is not None:
            self.p_threshold = p_threshold
        # filter out clusters smaller than that
        if min_cluster_size is not None:
            self.min_cluster_size = min_cluster_size
        # number of jobs for multiprocessing
        self.n_jobs = n_jobs if n_jobs <= self.N_CPUS else self.N_CPUS

        print('\nFinding ROIs ...')
        self.find_clusters()
        self.update_metadata()
        print('Analysis completed!')

        # print('Found %s cluster(s) with size:' % (self.cluster_count))
        # keys_cluster_sizes = sorted([key for key in self.metadata.keys() if
        #                              key.startswith('size_cluster_')])
        # for n, key in enumerate(keys_cluster_sizes):
        #     print(n + 1, ': %s voxels' % (self.metadata[key]))

        if output_path is not None:
            self.save_results(output_path)

    def draw_clusters(self, cubes=False):
        print("Drawing clusters ...")
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        y, z, x = np.where(~np.isnan(self.cluster_array))
        cluster_labels = list(np.sort(self.cluster_array[~np.isnan(self.cluster_array)]))
        if not cubes:
            ax.scatter(x, y, z, zdir='z', c=cluster_labels)
        elif cubes:
            cube_coords = list(np.vstack([x, y, z]).T)
            for c in cube_coords:
                self.draw_cube(c, ax, cluster_labels)

        ax.set_xlabel('Front (x)')
        ax.set_xlim(0, self.volume_shape[2])
        #ax.invert_xaxis()

        ax.set_ylabel('Side (y)')
        ax.set_ylim(0, self.volume_shape[0])

        ax.set_zlabel('Rows (z)')
        ax.set_zlim(0, self.volume_shape[1])
        ax.invert_zaxis()

    @staticmethod
    def draw_cube(coords, ax, cluster_labels):
        x, y, z = tuple(coords)
        x_span = [x, x + 1]
        y_span = [y, y + 1]
        z_span = [z, z - 1]
        for s, e in combinations(np.array(list(product(x_span, y_span, z_span))), 2):
            if np.sum(np.abs(s - e)) == 1:
                ax.plot(*zip(s, e), color='black')
        return ax

    def find_clusters(self):
        raise NotImplementedError

    def update_metadata(self):
        raise NotImplementedError

    def save_results(self):
        raise NotImplementedError


class SpectralClustering(ROIFinderBaseClass):
    def __init__(self, volumes=None, mask_img=None, random_seed=None):
        super().__init__(volumes, mask_img, random_seed)
        self.connectivity_matrix = None
        self.adjacency_matrix = None

    # def make_toy_data(self, dim=None):
    #     print('SubClass toy')

    def find_clusters(self):
        if self.n_jobs >= 1:
            # Define a spatial model
            ms = self.mask.shape
            # make a graph representation of the volume
            self.connectivity_matrix = grid_to_graph(ms[0], ms[1], ms[2],
                                                     self.mask).tocsr()
            ic, jc = self.connectivity_matrix.nonzero()
        # elif self.n_jobs > 1:
            # figure out multiprocessing if necessary

    def pearson_method(self, r_threshold=0.5):
        # According to Craddock et al. (2012)
        r_threshold = r_threshold
        n_voxels = len(self.voxel_1)
        v1_std = np.std(self.voxel_1)
        v1_mean = np.mean(self.voxel_1)
        v2_std = np.std(self.voxel_2)
        v2_mean = np.mean(self.voxel_2)
        # subtract mean from voxels
        v1_minus_mean = self.voxel_1 - v1_mean
        v2_minus_mean = self.voxel_2 - v2_mean
        # now sum product of arrays and divide by n-1 and std's
        r = np.sum(v1_minus_mean * v2_minus_mean) / ((n_voxels - 1) *
                                                     v1_std * v2_std)
        if r >= r_threshold:
            return r
        else:
            return 0

    def gaussian_similarity(self):
        # squared euclidean distance between two current voxels
        euclid_distance_square = np.exp(- (np.linalg.norm(self.voxel_1,
                                                          self.voxel_2)) ** 2)
        # mean squared euclidean distance across all!! neighbors
        # as in Thirion et al. (2014), but theres also other ways to determine
        # sigma_square
        sigma_square = np.mean(np.linalg.norm())
        return euclid_distance_square / (2 * sigma_square)


class PearsonMerger(ROIFinderBaseClass):
    def __init__(self, volumes=None, mask_img=None, random_seed=None):
        super().__init__(volumes, mask_img, random_seed)

    def find_clusters(self):
        if self.n_jobs >= 1:
            # iterates over all indices of the volume, returns tuples
            for index in np.ndindex(self.volume_shape):
                # check if element is not masked
                if not np.isnan(self.volumes[0][index]):
                    self.compute_clusters(index)
        # elif self.n_jobs > 1:
            # figure out multiprocessing if necessary

    def compute_clusters(self, center_index):
        # collect center voxel from all volumes
        center_voxels = self.get_voxels(center_index)
        # get neighboring voxel indices
        neighbor_indices = self.get_neighbor_indices(center_index)
        # extract voxel values for neighbors
        neighbor_voxels = self.get_neighbor_voxels(neighbor_indices)
        # Pearson with surrounding voxels
        correlation_scores = self.compute_correlation(center_voxels,
                                                      neighbor_voxels)
        # write changes to the cluster_array
        self.update_cluster_array(correlation_scores, center_index,
                                  neighbor_indices)

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
        return correlation_scores

    def make_correlation_histogram(self):
        pass

    def update_cluster_array(self, correlation_scores, center_idx,
                             neighbor_indices):
        # check how many values are stored in the correlation array and filter
        # out clusters that are smaller than the threshold
        if self.min_cluster_size is not None:
            # +1 for the center voxel
            if len(correlation_scores) + 1 >= self.min_cluster_size:
                self.update_cluster_util(correlation_scores, center_idx,
                                         neighbor_indices)
        # no cluster size filtering
        else:
            self.update_cluster_util(correlation_scores, center_idx,
                                     neighbor_indices)

    def update_cluster_util(self, correlation_scores, center_idx,
                            neighbor_indices):
        for key in correlation_scores.keys():
            index = tuple(neighbor_indices[key])
            # if no cluster is adjacent
            if not self.is_cluster_adjacent(index):
                self.cluster_count += 1
                self.cluster_array[index] = self.cluster_count
                self.cluster_array[center_idx] = self.cluster_count
            else:
                if self.cluster_count == 0:
                    self.cluster_count = 1
                self.cluster_array[index] = self.cluster_count
                self.cluster_array[center_idx] = self.cluster_count

    def is_cluster_adjacent(self, index):
        # get indices off ALL surrounding voxels
        all_neighbor_indices = self.get_neighbor_indices(index,
                                                         all_directions=True)
        # loop over all adjacent indices and check whether current voxel
        # is part of a larger cluster
        for val in all_neighbor_indices.values():
            # print(self.cluster_array[index])
            if not np.isnan(self.cluster_array[val]):
                return True
        return False

    def update_metadata(self):
        self.metadata['r_threshold'] = self.r_threshold
        self.metadata['p_threshold'] = self.p_threshold
        self.metadata['min_cluster_size'] = self.min_cluster_size
        self.metadata['cluster_count'] = self.cluster_count

        # loop over all clusters and get size
        for i in range(1, self.cluster_count + 1):
            cluster_id = "size_cluster_%d" % (i)
            cluster_size = np.where(self.cluster_array == i)[0].size
            self.metadata[cluster_id] = cluster_size

    def check_clusters_again():
        # control if some detected clusters need to be merged
        pass

    def save_results(self, output_path):
        timestamp = strftime('%d%m%y_%H%M', localtime())
        if self.p_threshold is not None:
            parameters = 'rt%s_pt%s_' % (self.r_threshold, self.p_threshold)
        else:
            parameters = 'rt%s_' % (self.r_threshold)
        file_name = parameters + timestamp + '.pickle'
        output_file = os.path.join(output_path, file_name)
        results = (self.metadata, self.cluster_array)
        with open(output_file, 'wb') as f:
            pickle.dump(results, f)
        print('Saved as %s' % (output_file))
