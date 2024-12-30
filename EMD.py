import ot 
import numpy as np
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cdist

"""
For a cell mask, find its four corner
"""
def corner_finading(array):
    rows, cols = np.where(array == 1)
    if len(rows) == 0 or len(cols) == 0:
        return None, None, None, None
    min_row = np.min(rows)
    max_row = np.max(rows)
    min_col = np.min(cols)
    max_col = np.max(cols)
    return min_row, max_row, min_col, max_col

"""
Extract the area of mask based on the four corner
"""
def extract_area(array, min_row, max_row, min_col, max_col):
    return array[min_row:max_row+1, min_col:max_col+1]

"""
Calculate the cost matrix for earthmover's distance
"""
def cost_matrix_calc(n_points):
    num = int(np.sqrt(n_points))
    cost_matrix = np.zeros((n_points, n_points))
    for i in range(n_points):
        for j in range(n_points):
            point1 = [i // num, i % num]
            point2 = [j // num, j % num]
            cost_matrix[i, j] = euclidean(point1, point2)
    return cost_matrix

"""
Standard earthmover's distance calculation, used for small masks
"""
def EMD_2d(array1, array2):
    union = np.logical_or(array1, array2).astype(np.int64)
    corners = corner_finading(union)
    array1 = extract_area(array1, *corners)
    array2 = extract_area(array2, *corners)
    m, n = array1.shape
    array1 = array1 / np.sum(array1)
    array2 = array2 / np.sum(array2)
    hist1_flat = array1.ravel().astype(np.float64)
    hist2_flat = array2.ravel().astype(np.float64)
    cost_matrix = cost_matrix_calc(len(hist1_flat))
    emd_distance = ot.sinkhorn2(hist1_flat,hist2_flat,cost_matrix)
    return emd_distance

"""
Sliced Wass distance, a faster EMD calculation for larger masks
"""
def fast_EMD(array1, array2):
    union = np.logical_or(array1, array2).astype(np.int64)
    corners = corner_finading(union)
    array1 = extract_area(array1, *corners)
    array2 = extract_area(array2, *corners)
    array1 = array1 / np.sum(array1)
    array2 = array2 / np.sum(array2)
    # 300 refers to the number of projection did in the Monte Carlo; You can change the number of projections
    return ot.sliced_wasserstein_distance(array1, array2, None, None, 10000)


