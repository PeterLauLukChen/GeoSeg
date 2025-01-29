import zarr

import CandidateSearching as cs
import numpy as np
import argparse
import os
import collections, functools, operator, pickle
from math import prod
import numpy as np
import networkx as nx
from scipy import ndimage

from scipy.ndimage import binary_dilation
from scipy.ndimage import label as ndi_label, binary_fill_holes
from scipy.ndimage import label as ndi_label
from scipy.ndimage import binary_fill_holes
from scipy.ndimage import binary_erosion

"""
    Generate a cell contact network graph from a segmented image and save it.

    Args:
        labels (np.array): 
            A 3D numpy array representing the segmented image, where each 
            unique integer corresponds to a labeled segment (e.g., a cell).
        graph_path (str): 
            Path to save the generated graph object. The graph is saved in 
            pickle format.
        anisotropy (tuple, optional): 
            A tuple representing the voxel anisotropy in the (z, y, x) 
            dimensions. Defaults to (.4, .1, .1).
        bit_shift (int, optional): 
            The number of bits to shift for encoding and decoding contact pairs. 
            Defaults to 32.

    Returns:
        None: 
            The function saves the graph to the specified file path but does 
            not return anything.

    Notes:
        - The `weight` attribute of each edge in the graph represents the 
          contact surface area between two connected nodes.
    """
def generate_graph_from_mask(
    labels: np.array, 
    graph_path: str, 
    anisotropy: tuple = (.4, .1, .1), 
    bit_shift: int = 32
) -> None:
    nodes, voxels = np.unique(labels, return_counts=True)
    voxel_size = prod(anisotropy)
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    img = labels.astype('int64')
    r = [np.roll(img, 1, dim) for dim in range(0, 3)]
    e = [np.array(np.minimum((img << bit_shift) | r[dim], img | (r[dim] << bit_shift)))[np.s_[int(dim == 0):, int(dim == 1):, int(dim == 2):]] for dim in range(0, 3)]
    contacts = [np.unique(e[dim], return_counts=True) for dim in range(0, 3)]
    contacts = [dict(zip(contacts[dim][0], contacts[dim][1] * anisotropy[dim])) for dim in range(0, 3)]
    contacts = dict(functools.reduce(operator.add, map(collections.Counter, contacts)))
    for contact, contact_surface_area in contacts.items():
        i = contact >> bit_shift
        j = contact & ((1 << bit_shift) - 1)
        if i != j:
            graph.add_edge(i, j, weight=contact_surface_area)
    with open(graph_path, 'wb') as f:
        pickle.dump(graph, f)

def tiltedseg(cell_ids, penalty, mode):
    try:
        # Load data and 2D segmention layers
        image_path = "./data"
        saved_path = "./temp.npy"
        os.makedirs(saved_path, exist_ok=True)

        zarr_image = zarr.open(image_path, mode="r")
        segmentation = np.array(zarr_image)
        highlighted_cells = np.zeros_like(segmentation, dtype=np.uint8)
        for label, cell_id in enumerate(cell_ids, start=1):
            highlighted_cells[segmentation == cell_id] = label

        # Step 1: Finding the connecting curvature between cells
        cell_1_mask = highlighted_cells == 1
        cell_2_mask = highlighted_cells == 2
        boundary_1 = binary_dilation(cell_1_mask) & cell_2_mask
        boundary_2 = binary_dilation(cell_2_mask) & cell_1_mask
        connecting_surface = boundary_1 | boundary_2
        surface_coords = np.argwhere(connecting_surface)
        curvature_layer = np.zeros_like(segmentation, dtype=np.uint8)

        curvature_layer = np.zeros_like(segmentation, dtype=np.uint8)
        for coord in surface_coords:
            x, y, z = coord
            curvature_layer[x, y, z] = 1

        # Step 2: Fit the curvature via PCA plane
        x = surface_coords[:, 0]
        y = surface_coords[:, 1]
        z = surface_coords[:, 2]

        coords = surface_coords - surface_coords.mean(axis=0)
        cov_matrix = np.cov(coords.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix) # Retrieving the eigenvecs to construct the plane
        normal_vector = eigenvectors[:, np.argmin(eigenvalues)]
        plane_point = surface_coords.mean(axis=0)

        x_range = np.arange(0, segmentation.shape[0])
        y_range = np.arange(0, segmentation.shape[1])
        yy, xx = np.meshgrid(y_range, x_range)

        n1, n2, n3 = normal_vector
        x0, y0, z0 = plane_point
        zz = -(n1 * (xx - x0) + n2 * (yy - y0)) / n3 + z0 # Reconvering the plane given normal vec and a point the plane crossed through
        fitted_plane_layer = np.zeros_like(segmentation, dtype=np.uint8)

        # For visualization purpose
        thickness = 1 
        for z in range(segmentation.shape[2]):
            mask = (np.abs(zz - z) <= thickness)
            if np.any(mask): 
                fitted_plane_layer[xx[mask], yy[mask], z] = 2
        highlighted_cells = np.zeros_like(segmentation, dtype=np.uint8)
        for label, cell_id in enumerate(cell_ids, start=1):
            highlighted_cells[segmentation == cell_id] = label

        # Step 3: Using new plane to cut the cell and rotate the cell
        cell_id_to_label = {cell_ids[0]: 1, cell_ids[1]: 2}
        cell_mask = np.isin(segmentation, cell_ids)
        cell_coords = np.argwhere(cell_mask)
        projections = np.dot(cell_coords - plane_point, normal_vector)
        min_proj = projections.min()
        max_proj = projections.max()

        padding = 5.0 # Preparation to slice the cell layer by layer to reconstruct the rotated cell
        step_size = 1.0  
        tolerance = 0.5  
        steps = np.arange(min_proj - padding, max_proj + padding, step_size)
        if np.allclose(normal_vector, [0, 0, 1]) or np.allclose(normal_vector, [0, 0, -1]):
            u = np.array([1, 0, 0])
        else:
            u = np.cross(normal_vector, [0, 0, 1])
            u /= np.linalg.norm(u)
        v = np.cross(normal_vector, u)
        v /= np.linalg.norm(v)

        s_all = []
        t_all = []
        slice_indices = []
        labels_all = []

        for idx, step in enumerate(steps):
            shifted_plane_point = plane_point + step * normal_vector
            distances = np.dot(cell_coords - shifted_plane_point, normal_vector)
            mask = np.abs(distances) <= tolerance

            intersected_points = cell_coords[mask]
            if intersected_points.size == 0:
                continue
            p = intersected_points - shifted_plane_point
            s = np.dot(p, u)
            t = np.dot(p, v)
            s_all.extend(s)
            t_all.extend(t)
            slice_indices.extend([idx] * len(s))
            original_labels = segmentation[intersected_points[:, 0],
                                        intersected_points[:, 1],
                                        intersected_points[:, 2]]
            mapped_labels = [cell_id_to_label.get(label, 0) for label in original_labels]
            labels_all.extend(mapped_labels)

        s_all = np.array(s_all)
        t_all = np.array(t_all)
        slice_indices = np.array(slice_indices)
        labels_all = np.array(labels_all)
        min_s = s_all.min()
        max_s = s_all.max()
        min_t = t_all.min()
        max_t = t_all.max()
        pixel_size = 1.0 

        s_bins = int(np.ceil((max_s - min_s) / pixel_size)) + 1
        t_bins = int(np.ceil((max_t - min_t) / pixel_size)) + 1
        num_slices = len(steps)
        image_stack = np.zeros((num_slices, s_bins, t_bins), dtype=np.uint8)
        s_indices = np.floor((s_all - min_s) / pixel_size).astype(int)
        t_indices = np.floor((t_all - min_t) / pixel_size).astype(int)
        for s_idx, t_idx, slice_idx, label in zip(s_indices, t_indices, slice_indices, labels_all):
            image_stack[slice_idx, s_idx, t_idx] = label

         # Iterate through slices
        for idx in range(image_stack.shape[0]):
            slice = image_stack[idx, :, :]
            for label_value in [1, 2]:
                label_mask = (slice == label_value)
                if np.any(label_mask):
                    labeled_array, num_features = ndi_label(label_mask, structure=np.ones((3,3)))
                    for component_idx in range(1, num_features + 1):
                        component_mask = (labeled_array == component_idx)
                        filled_component = binary_fill_holes(component_mask) # 2nd interpolation to fill holes at the boundary
                        slice[filled_component & ~component_mask] = label_value 
            image_stack[idx, :, :] = slice

        cnt = 0
        for idx in range(num_slices):
            slice = image_stack[idx, :, :]
            if np.any(slice == 1) and np.any(slice == 2):
                cnt += 1
        
        if cnt > 18:
            return

        # Step 4: create the gap to prepare it to the pre-trained model
        normal_vector = normal_vector / np.linalg.norm(normal_vector)
        cell_id_to_label = {cell_ids[0]: 1, cell_ids[1]: 2}
        plane_layer_index = np.argmin(np.abs(steps))

        plane_slice = image_stack[plane_layer_index, :, :]
        area_i = np.sum(plane_slice > 0)

        plane_slice = image_stack[plane_layer_index + 1, :, :]
        area_j = np.sum(plane_slice > 0)

        plane_slice = image_stack[plane_layer_index - 1, :, :]
        area_k = np.sum(plane_slice > 0)

        if abs(area_i - area_k) / min(area_i, area_k) > 0.1:
            return
        
        if abs(area_i - area_j) / min(area_i, area_j) > 0.1:
            return

        image_stack[plane_layer_index, :, :] = 0

        for idx in range(plane_layer_index + 1, num_slices):
            slice = image_stack[idx, :, :]
            slice[slice > 0] = 1
            image_stack[idx, :, :] = slice
        for idx in range(0, plane_layer_index):
            slice = image_stack[idx, :, :]
            slice[slice > 0] = 2
            image_stack[idx, :, :] = slice
        np.save(saved_path, image_stack)

        filepath = "./temp.npy"
        result = cs.main(filepath, penalty, mode)
        
        if len(result) == 0:
            return 
        else:
            print(cell_ids)
        
    except Exception as e:
        print(f"Error processing cell pair {cell_ids}: {e}")

def main(penalty, mode):
    zarr_path = "./data"
    zarr_data = zarr.open(zarr_path, mode='r')
    segmented_labels = zarr_data[:] 
    output_path = "./contact_graph.pkl"
    generate_graph_from_mask(labels=segmented_labels, graph_path=output_path)

    graph_path = "./contact_graph.pkl"
    with open(graph_path, 'rb') as f:
        graph = pickle.load(f)
    edges = list(graph.edges())
    connections_array = np.array(edges)
    np.save("./neighbor.npy", connections_array)

    cell_neighbors = np.load("./neighbor.npy")
    for cell_pair in cell_neighbors:
        tiltedseg(cell_pair, penalty, mode)

parser = argparse.ArgumentParser()
parser.add_argument('--PENALTY', type=float, default='1.3', help='Penalty Applied to the tolerance in the CandidateSearching')
parser.add_argument('--MODE', type=str, default='minmax', help='minmax or q1q3 used in CandidateSearching')

args = parser.parse_args()
penalty = args.PENALTY
mode = args.MODE
main(penalty, mode)

