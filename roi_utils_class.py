import os
from itertools import combinations, product
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr


class ROIFinder:
    # toy data specifics
    TOY_DIM = 10  # cube size
    N_TOY_SUBJECTS = 5
    TOY_LOW = - 10
    TOY_UP = abs(TOY_LOW)
    # 3d directions to find neighbor voxels
    DIRECTIONS = dict(front=np.array([-1, 0, 0]), back=np.array([1, 0, 0]),
                      top=np.array([0, -1, 0]), bottom=np.array([0, 1, 0]),
                      left=np.array([0, 0, -1]), right=np.array([0, 0, 1]))

    # correlation thresholds
    R_THRESHOLD = 0.5
    P_TRESHOLD = 0.005
    # number of CPUs for multiprocessing
    N_CPUS = os.cpu_count()

    def __init__(self, volumes=None, r_threshold=R_THRESHOLD,
                 p_threshold=P_TRESHOLD, n_jobs=1):
        if volumes is not None:
            # check if volumes are provided as 4d array or list of 3d arrays
            if isinstance(volumes, np.ndarray):
                n_subjects = volumes.shape[0]
                self.volumes = [volumes[s, ...] for s in range(n_subjects)]
            elif isinstance(volumes, list):
                self.volumes = volumes
        elif volumes is None:
            self.volumes = self._make_toy_data()

        self.volume_shape = self.volumes[0].shape
        self.r_threshold = r_threshold
        self.p_threshold = p_threshold
        self.n_jobs = n_jobs if n_jobs <= self.N_CPUS else self.N_CPUS

    @staticmethod
    def _make_toy_data(dim=TOY_DIM, n_subjects=N_TOY_SUBJECTS,
                       lower_bound=TOY_LOW, upper_bound=TOY_UP):
        # random volumes for testing
        low, up = lower_bound, upper_bound
        toy_volumes = [(up - low) * np.random.random_sample(tuple([dim] * 3)) +
                       low for _ in range(n_subjects)]
        return toy_volumes

    def _compute_correlation(self, center_voxels, neighbors_voxels):
        pearson_scores = dict()
        for key, val in neighbors_voxels.items():
            r, p = pearsonr(val, center_voxels)
            if (abs(r) > self.r_threshold) and (p < self.p_threshold):
                pearson_scores[key] = r, p
        return pearson_scores

    def _get_neighbor_indices(self, center_index):
        center = list(center_index)
        neighbor_indices = dict()
        for key, val in self.DIRECTIONS.items():
            coord = np.array(center) + val
            coord = self.__index_check(coord)
            if coord is not None:
                neighbor_indices[key] = coord
        return neighbor_indices

    def __index_check(self, indices):
        idx_list = list(indices)
        for i, n in enumerate(idx_list):
            if (n < 0) or (n > self.volume_shape[i] - 1):
                return None
        return indices

    def _get_neighbor_voxels(self, indices):
        neighbor_voxels = dict()
        for key, val in indices.items():
            index = tuple(val)
            voxels = [v[index] for v in self.volumes]
            neighbor_voxels[key] = np.array(voxels)
        return neighbor_voxels

    def update_cluster_array(correlation_scores, center_idx, neighbors_indices,
                             cluster_array):
        for key, val in correlation_scores.items():
            idx = tuple(neighbors_indices[key])
            r, p = val
            # todo --> mark individual clusters with different integers
            cluster_array[idx] = 1
            if cluster_array[center_idx] == 0:
                cluster_array[center_idx] = 1
        return cluster_array

    def draw_clusters(cluster_array, ax=None, cubes=False):
        if ax is None:
            ax = plt.gca()
        else:
            ax = ax
        x, z, y = cluster_array.nonzero()
        if not cubes:
            ax.scatter(x, y, z, zdir='z', c='red')
        elif cubes:
            cube_coords = list(np.vstack([x, y, z]).T)
            for c in cube_coords:
                __draw_cube(c, ax=ax)
        return ax

    # draw cube
    def __draw_cube(coords, ax):
        x, y, z = tuple(coords)
        x_span = [x, x + 1]
        y_span = [y, y + 1]
        z_span = [z, z - 1]
        for s, e in combinations(np.array(list(product(x_span,
                                                       y_span,
                                                       z_span))), 2):
            if np.sum(np.abs(s - e)) == 1:
                ax.plot(*zip(s, e), color="black")
        return ax


    def find_rois(it, volumes, volume_shape, cluster_array, rt=0.5, pt=0.05):
        # collect center voxel from all volumes
        center_idx = it.multi_index
        center_voxels = np.array([v[center_idx] for v in volumes])
        # get neighboring voxel indices
        neighbors_indices = get_neighbor_indices(center_idx, volume_shape)
        # extract voxel values for neighbors
        neighbors_voxels = get_neighbor_voxels(neighbors_indices, volumes)
        # Pearson with surrounding voxels
        correlation_scores = compute_correlation(center_voxels, neighbors_voxels,
                                                 rt, pt)
        # array with integers for individual clusters
        cluster_array = update_cluster_array(correlation_scores, center_idx,
                                             neighbors_indices, cluster_array)
        return cluster_array
