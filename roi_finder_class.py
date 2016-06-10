import os
from itertools import combinations, product
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
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

    def __init__(self, volumes=None, r_threshold=0.5,
                 p_threshold=None, n_jobs=1, min_cluster_size=None,
                 random_seed=None):
        # set random seed if needed
        if random_seed is not None:
                np.random.seed(random_seed)
        # check dtype of provided volumes, generate toy data if needed
        if volumes is not None:
            # check if volumes are provided as 4d array or list of 3d arrays
            if isinstance(volumes, np.ndarray):
                n_subjects = volumes.shape[0]
                self.volumes = [volumes[s, ...] for s in range(n_subjects)]
            elif isinstance(volumes, list):
                self.volumes = volumes
            elif isinstance(volumes, int):
                self.volumes = self.make_toy_data(dim=volumes)
        elif volumes is None:
            self.volumes = self.make_toy_data()
        # shape of the first volume, assumes all volumes have same shape
        self.volume_shape = self.volumes[0].shape
        # number of subjects
        self.n_subjects = self.volume_shape[0]
        # arbitrary threshold for pearson coefficient
        self.r_threshold = r_threshold
        # arbitrary threshold for p-value of pearson test
        self.p_threshold = p_threshold
        # filter out clusters smaller than that
        self.min_cluster_size = min_cluster_size
        # keeps track of number of different clusters
        self.cluster_count = 0
        # array that contains integers that indicate clusters
        self.cluster_array = np.zeros(self.volume_shape, dtype=int)
        # number of jobs for multiprocessing
        self.n_jobs = n_jobs if n_jobs <= self.N_CPUS else self.N_CPUS

    @staticmethod
    def make_toy_data(dim=TOY_DIM, n_subjects=N_TOY_SUBJECTS,
                      lower_bound=TOY_LOW, upper_bound=TOY_UP):
        # random volumes for testing
        low, up = lower_bound, upper_bound
        toy_volumes = [(up - low) * np.random.random_sample(tuple([dim] * 3)) +
                       low for _ in range(n_subjects)]
        return toy_volumes

    def get_neighbor_indices(self, center_index, all_directions=False):
        center = list(center_index)
        neighbor_indices = dict()
        directions = (self.DIRECTIONS if all_directions is False
                      else self.ALL_DIRECTIONS)
        for key, val in directions.items():
            coord = np.array(center) + val
            coord = self._index_check(coord)
            if coord is not None:
                neighbor_indices[key] = coord
        # returns dict with valid indices of adjacent cells
        return neighbor_indices

    def _index_check(self, indices):
        idx_list = list(indices)
        for i, n in enumerate(idx_list):
            if (n < 0) or (n > self.volume_shape[i] - 1):
                return None
        return indices

    def get_neighbor_voxels(self, indices):
        neighbor_voxels = dict()
        for key, val in indices.items():
            index = tuple(val)
            neighbor_voxels[key] = self.get_voxels(index)
        return neighbor_voxels

    def get_voxels(self, index):
        return np.array([v[index] for v in self.volumes])

    def compute_correlation(self, center_voxels, neighbor_voxels):
        correlation_scores = dict()
        for key, val in neighbor_voxels.items():
            r, p = pearsonr(center_voxels, val)
            # filter according to threshold levels for r and p
            # p values are not entirely reliable according to documentation
            # only for datasets larger than 500
            if abs(r) > self.r_threshold:
                if self.p_threshold is not None:
                    if p <= self.p_threshold:
                        correlation_scores[key] = r, p
                elif self.p_threshold is None:
                    correlation_scores[key] = r, p
        return correlation_scores

    def update_cluster_array(self, correlation_scores, center_idx,
                             neighbor_indices):
        # check how many values are stored in the correlation array and filter
        # out clusters that are smaller than the threshold
        if self.min_cluster_size is not None:
            # +1 for the center voxel
            if len(correlation_scores) + 1 >= self.min_cluster_size:
                self._update_cluster_util(correlation_scores, center_idx,
                                          neighbor_indices)
        # no cluster size filtering
        elif self.min_cluster_size is None:
            self._update_cluster_util(correlation_scores, center_idx,
                                      neighbor_indices)

    def _update_cluster_util(self, correlation_scores, center_idx,
                             neighbor_indices):
        for key in correlation_scores.keys():
            index = tuple(neighbor_indices[key])
            # if no clusters are adjacent
            if not self._check_adjacent_clusters(index):
                self.cluster_count += 1
                self.cluster_array[index] = self.cluster_count
                self.cluster_array[center_idx] = self.cluster_count
            else:
                if self.cluster_count == 0:
                    self.cluster_count = 1
                self.cluster_array[index] = self.cluster_count
                self.cluster_array[center_idx] = self.cluster_count

    def _check_adjacent_clusters(self, index):
        # get indices off ALL surrounding voxels
        all_neighbor_indices = self.get_neighbor_indices(index, all_directions=True)
        # loop over all adjacent indices and check whether current voxel
        # is part of a larger cluster
        for val in all_neighbor_indices.values():
            index = tuple(val)
            # print(self.cluster_array[index])
            if self.cluster_array[index] != 0:
                return True
        return False

    def find_clusters(self):
        if self.n_jobs >= 1:
            for index in np.ndindex(self.volume_shape):
                self._compute_clusters(index)
        # figure out multiprocessing if necessary
        # doesnt work as it is now
        # elif self.n_jobs > 1:
            # try:
            #     pool = Pool(processes=self.n_jobs)
            #     pool.map(self._compute_clusters, [index for index in
            #                                   np.ndindex(self.volume_shape)])
            # finally:
            #     pool.close()
            #     pool.join()

    def _compute_clusters(self, center_index):
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

    def draw_clusters(self, ax=None, cubes=False):
        if ax is None:
            ax = plt.gca()
        else:
            ax = ax
        x, z, y = self.cluster_array.nonzero()
        cluster_labels = list(np.sort(self.cluster_array[np.nonzero(self.cluster_array)]))
        if not cubes:
            ax.scatter(x, y, z, zdir='z', c=cluster_labels)
        elif cubes:
            cube_coords = list(np.vstack([x, y, z]).T)
            for c in cube_coords:
                self._draw_cube(c, ax, cluster_labels)
        return ax

    @staticmethod
    def _draw_cube(coords, ax, cluster_labels):
        x, y, z = tuple(coords)
        x_span = [x, x + 1]
        y_span = [y, y + 1]
        z_span = [z, z - 1]
        for s, e in combinations(np.array(list(product(x_span, y_span, z_span))), 2):
            if np.sum(np.abs(s - e)) == 1:
                ax.plot(*zip(s, e), color=cluster_labels)
        return ax

if __name__ == '__main__':
    from timeit import default_timer as timer

    dim = 10
    RF = ROIFinder(volumes=dim, r_threshold=0.8, min_cluster_size=4)
    # start = timer()
    clusters = RF.find_clusters()
    # end = timer()
    # print((end - start))
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    RF.draw_clusters(ax=ax, cubes=False)
    ax.set_xlabel('Depth (x)')
    ax.set_xlim(0, dim)
    ax.invert_xaxis()

    ax.set_ylabel('Columns (y)')
    ax.set_ylim(0, dim)

    ax.set_zlabel('Rows (z)')
    ax.set_zlim(0, dim)
    ax.invert_zaxis()
    print(RF.cluster_array)
