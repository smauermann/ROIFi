import os
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from roi_utils import (compute_correlation, draw_clusters, find_rois,
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
# multiprocessing
pool = Pool(processes=os.cpu_count())
cluster_array = pool.map(find_rois(it, volumes, volume_shape, cluster_array,
                                   rt=CORR_THRESHOLD, pt=P_TRESHOLD))
# loops over all elements, in a memory-efficient manner, no particular order


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
