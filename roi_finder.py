import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from roi_utils import (compute_correlation, draw_clusters,
                       get_neighbor_indices, get_neighbor_voxels,
                       update_cluster_array)

# toy data specifics
DIM = 10  # cube size
N_DIMENSIONS = 3
N_CUBES = 5
# correlation thresholds
CORR_THRESHOLD = 0.5
P_TRESHOLD = 0.005

# random volumes for testing
lower_bound = -10
upper_bound = abs(lower_bound)
volumes = [(upper_bound - lower_bound) *
           np.random.random_sample((DIM, DIM, DIM)) +
           lower_bound for _ in range(N_CUBES)]
volume_shape = volumes[0].shape
cluster_array = np.zeros(volume_shape, dtype=int)
# iterator object
it = np.nditer(volumes[0], flags=['multi_index'])
# loops over all elements, in a memory-efficient manner, no particular order
for element in it:
    # collect center voxel from all volumes
    center_idx = it.multi_index
    center_voxels = np.array([v[center_idx] for v in volumes])
    # get neighboring voxel indices
    neighbors_indices = get_neighbor_indices(center_idx, volume_shape)
    # extract voxel values for neighbors
    neighbors_voxels = get_neighbor_voxels(neighbors_indices, volumes)
    # Pearson with surrounding voxels
    correlation_scores = compute_correlation(center_voxels, neighbors_voxels,
                                             CORR_THRESHOLD, P_TRESHOLD)
    # array with integers for individual clusters
    cluster_array = update_cluster_array(correlation_scores, center_idx,
                                         neighbors_indices, cluster_array)


fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
draw_clusters(cluster_array, ax=ax, cubes=True)

ax.set_xlabel('Depth (x)')
ax.set_xlim(0, DIM)
ax.invert_xaxis()

ax.set_ylabel('Columns (y)')
ax.set_ylim(0, DIM)

ax.set_zlabel('Rows (z)')
ax.set_zlim(0, DIM)
ax.invert_zaxis()



    # if correlation_scores:
    #     for key, val in correlation_scores.items():
    #         print(tuple(neighbors_indices[key]))
    #     print(center_idx)
    #     print(correlation_scores)
    #     print(cluster_array)
    # indicate clusters in cluster_array
