from time import localtime, strftime

import numpy as np
from scipy.stats import pearsonr
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.manifold import spectral_embedding

from roi_finder_base import ROIFinderBase

try:
    import cPickle as pickle
except:
    import pickle

class SpectralClustering(ROIFinderBase):
    def __init__(self, volumes=None, mask_img=None, random_seed=None):
        super().__init__(volumes, mask_img, random_seed)
        self.connectivity_matrix = None
        self.adjacency_matrix = None

    def make_toy_data(self, dim):
        print('SubClass toy')

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


class PearsonMerger(ROIFinderBase):
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
