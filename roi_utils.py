from itertools import combinations, product

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

# 3d directions
DIRECTIONS = dict(front=np.array([-1, 0, 0]), back=np.array([1, 0, 0]),
                  top=np.array([0, -1, 0]), bottom=np.array([0, 1, 0]),
                  left=np.array([0, 0, -1]), right=np.array([0, 0, 1]))


def compute_correlation(center_voxels, neighbors_voxels, rt=0.5, pt=0.05):
    pearson_scores = dict()
    for key, val in neighbors_voxels.items():
        r, p = pearsonr(val, center_voxels)
        if (abs(r) > rt) and (p < pt):
            pearson_scores[key] = r, p
    return pearson_scores


def get_neighbor_indices(center_index, volume_shape):
    center = list(center_index)
    neighbor_indices = dict()
    for key, val in DIRECTIONS.items():
        coord = np.array(center) + val
        coord = _index_check(coord, volume_shape)
        if coord is not None:
            neighbor_indices[key] = coord
    return neighbor_indices


def _index_check(indices, volume_shape):
    idx_list = list(indices)
    for i, n in enumerate(idx_list):
        if (n < 0) or (n > volume_shape[i] - 1):
            return None
    return indices


def get_neighbor_voxels(indices, volumes):
    neighbor_voxels = dict()
    for key, val in indices.items():
        index = tuple(val)
        voxels = [v[index] for v in volumes]
        neighbor_voxels[key] = np.array(voxels)
    return neighbor_voxels


def update_cluster_array(correlation_scores, center_idx, neighbors_indices,
                         cluster_array):
    for key, val in correlation_scores.items():
        idx = tuple(neighbors_indices[key])
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
            _draw_cube(c, ax=ax)
    return ax


# draw cube
def _draw_cube(coords, ax):
    x, y, z = tuple(coords)
    x_span = [x, x + 1]
    y_span = [y, y + 1]
    z_span = [z, z - 1]
    for s, e in combinations(np.array(list(product(x_span, y_span, z_span))), 2):
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
