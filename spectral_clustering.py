import numpy as np


# raw script for spectral clustering
class SpectralClustering:
    def __init__(self, voxel_1, voxel_2, method=None):
        self.voxel_1 = voxel_1
        self.voxel_2 = voxel_2

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
